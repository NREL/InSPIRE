import pvdeg
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask import delayed, compute
import os
import gc
import xarray as xr
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO)

WEATHER_DB = "NSRDB"
WEATHER_ARG = {
    "satellite": "Americas",
    "names": "TMY",
    "NREL_HPC": True,
    "attributes": [
        "air_temperature",
        "wind_speed",
        "dhi",
        "ghi",
        "dni",
        "relative_humidity",
    ],
}

@delayed
def combine_many_nc_to_zarr(conf: str, target_dir:Path) -> None:
    """combine files for a single config"""
    logging.info(f"merging files for {conf}")
    combined_ds = xr.open_mfdataset(f"{target_dir}/{conf}/*.nc")
    combined_ds.to_zarr(f"{target_dir}/{conf}-pvdeg-pysam", consolidated=True)


def get_local_weather(local_test_paths: dict):
    if "weather" not in local_test_paths:
        raise ValueError('"local_test_paths" must contain key weather with value as file path')
    if "meta" not in local_test_paths:
        raise ValueError('"local_test_paths" must contain key meta with value as file path')

    # need to be tmy files
    geo_weather = xr.open_dataset(local_test_paths["weather"])
    geo_meta = pd.read_csv(local_test_paths["meta"], index_col=0)

    if "wind_direction" not in geo_weather.data_vars:
        logging.info("using placeholder wind direction, filling zeros")
        geo_weather = geo_weather.assign(wind_direction=geo_weather["temp_air"] * 0)
    
    if "albedo" not in geo_weather.data_vars:
        logging.info("using placeholder wind direction, filling 0.2 for all timesteps")
        geo_weather = geo_weather.assign(albedo=geo_weather["temp_air"] * 0 + 0.2) 

    logging.info(f"{len(geo_meta)} location entries in local data")

    # could move this to the analysis
    geo_weather['time'] = pd.date_range(start="2001-01-01 00:30:00", periods=8760, freq='1h')
    chunk_size = 10

    return geo_weather, geo_meta, chunk_size


@delayed
def process_slice(
    geo_weather: xr.Dataset,
    geo_meta: pd.DataFrame,
    conf: str,
    conf_dir: str,
    target_dir: str,
    front: int,
    back: int,
):
    # we may want to force slice_weather.index to be the same as template.index
    # they are tmy and in the middle of the hour already but could have different years
    # tmy so year does not matter, it is just there as a placeholder

    slice_weather = geo_weather.isel(gid=slice(front, back))
    slice_meta = geo_meta.iloc[front:back]

    template = pvdeg.geospatial.output_template(
        ds_gids=slice_weather,
        shapes=pvdeg.pysam.INSPIRE_GEOSPATIAL_TEMPLATE_SHAPES,
        add_dims={'distance':10}
    )

    partial_res = pvdeg.geospatial.analysis(
        weather_ds=slice_weather,
        meta_df=slice_meta,

        func=pvdeg.pysam.inspire_ground_irradiance,
        template=template,
        config_files={
            'pv' : f"{conf_dir}/{conf}/{conf}_pvsamv1.json"
        }
    )

    # ideally we write everything to the same zarr store but need to determine chunking and dimensions
    # partial_res.to_zarr(
    #     conf_zarr_path, 
    #     region="auto", 
    #     compute=True, 
    #     mode="a", 
    #     safe_chunks=True
    # )

    fname = Path(f"{target_dir}/{conf}/{front}-{back}.nc")
    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    partial_res.to_netcdf(fname)
    logging.info(f"saved to file | {fname.resolve()}")

def run_state(
    state: str,
    target_dir: Path,
    conf_dir: Path,
    dask_client,
    confs=[
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
    ],
    local_test_paths: dict = None,
) -> None:

    logging.info(local_test_paths)

    # created for local test files
    if local_test_paths is not None:
        # get local weather
        geo_weather, geo_meta, chunk_size = get_local_weather(local_test_paths=local_test_paths)

    else:
        # get NSRDB
        geo_weather, geo_meta = pvdeg.weather.get(
            WEATHER_DB, geospatial=True, **WEATHER_ARG
        )

        # downselect NSRDB
        geo_meta = geo_meta[geo_meta["country"] == "United States"]
        geo_meta = geo_meta[geo_meta["state"] == state]
        
        chunk_size = 40

    # downselect and chunk
    geo_weather = geo_weather.sel(gid=geo_meta.index)
    geo_weather = geo_weather.chunk({"gid": chunk_size})

    results = [] # delayed objects list

    # chunk size of 10, use 16 workers for this default (run as substeps so it can be handled in memory)
    # build the delayed graph
    step_size = 640  
    for conf in confs:

        # # initialize full zarr store
        # config_zarr_path = f"{target_dir}/{conf}-pvdeg-pysam"
        # full_output = pvdeg.geospatial.output_template(
        #     ds_gids=geo_weather,
        #     shapes=pvdeg.pysam.INSPIRE_GEOSPATIAL_TEMPLATE_SHAPES,
        #     add_dims={'distance':10}
        # )

        # full_output.to_zarr(config_zarr_path, compute=False, mode='w')

        # create delayed objects for each slice
        for i in range(0, len(geo_meta), step_size):
            front = i
            back = min(i + step_size, len(geo_meta) - 1)

            results.append(
                process_slice(
                    geo_weather=geo_weather,
                    geo_meta=geo_meta,
                    conf=conf,
                    conf_dir=conf_dir,
                    target_dir=target_dir,
                    front=front,
                    back=back,
                )
            )

    writes = [combine_many_nc_to_zarr(conf=conf, target_dir=target_dir) for conf in confs]

    batch_size = 2 # do two chunks at a time
    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]

        with ProgressBar():
            compute(*batch)

        # force garbage collect on workers
        dask_client.run(gc.collect)

    with ProgressBar():
        compute(*writes)
    
    return