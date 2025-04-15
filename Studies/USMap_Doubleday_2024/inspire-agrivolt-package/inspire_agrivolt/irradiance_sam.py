import pvdeg
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask import delayed, compute
import os
import gc
import xarray as xr
import pandas as pd
import time
import numpy as np

from inspire_agrivolt import logger

WEATHER_DB = "NSRDB"
WEATHER_ARG = {
    "satellite": "Americas",
    "names": "TMY",
    "NREL_HPC": True,
    "attributes": pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
}

@delayed
def combine_many_nc_to_zarr(conf: str, target_dir:Path) -> None:
    """combine files for a single config"""
    logger.info(f"merging files for {conf}")
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
        logger.info("using placeholder wind direction, filling zeros")
        geo_weather = geo_weather.assign(wind_direction=geo_weather["temp_air"] * 0)
    
    if "albedo" not in geo_weather.data_vars:
        logger.info("using placeholder wind direction, filling 0.2 for all timesteps")
        geo_weather = geo_weather.assign(albedo=geo_weather["temp_air"] * 0 + 0.2) 

    logger.info(f"{len(geo_meta)} location entries in local data")

    # could move this to the analysis
    geo_weather['time'] = pd.date_range(start="2001-01-01 00:30:00", periods=8760, freq='1h')
    chunk_size = 10

    return geo_weather, geo_meta, chunk_size

def load_weather(
    local_test_paths: dict, 
    state: str 
) -> tuple[xr.Dataset, pd.DataFrame]:
    if local_test_paths is not None:
        logger.info("LOADING: weather dataset from local files")
        geo_weather, geo_meta, chunk_size = get_local_weather(local_test_paths=local_test_paths)

    else:
        logger.info("LOADING: weather dataset from NSRDB on kestrel")
        start = time.time()

        geo_weather, geo_meta = pvdeg.weather.get(
            WEATHER_DB, geospatial=True, **WEATHER_ARG
        )

        end = time.time()
        logger.info(f"LOADED: dataset in: {end - start} s")

        # downselect NSRDB
        logger.info(f"DOWNSELECTING: keeping points where 'state'== '{state}'")
        start = time.time()
        geo_meta = geo_meta[geo_meta["country"] == "United States"]
        geo_meta = geo_meta[geo_meta["state"] == state]
        end = time.time()
        logger.info(f"DOWNSELECTING: took {end - start} seconds")
        
        chunk_size = 40

    geo_weather = geo_weather.sel(gid=geo_meta.index).chunk({"gid": chunk_size})

    return geo_weather, geo_meta, chunk_size




@delayed
def process_slice(
    conf: str,
    conf_dir: str,
    target_dir: str,
    sub_gids: np.ndarray,
    state:str,
    local_test_paths: dict=None,
    # geo_weather: xr.Dataset,
    # geo_meta: pd.DataFrame,
    # front: int,
    # back: int,
):
    # we may want to force slice_weather.index to be the same as template.index
    # they are tmy and in the middle of the hour already but could have different years
    # tmy so year does not matter, it is just there as a placeholder

    geo_weather, geo_meta, chunk_size = load_weather(local_test_paths=local_test_paths, state=state)

    slice_weather = geo_weather.sel(gid=sub_gids).chunk({"gid": chunk_size})
    slice_meta = geo_meta.loc[sub_gids]

    # update variable names to match convention
    slice_weather = pvdeg.weather.map_weather(slice_weather)

    # slice_weather = geo_weather.isel(gid=slice(front, back))
    # slice_meta = geo_meta.iloc[front:back]

    slice_template = pvdeg.geospatial.output_template(
        ds_gids=slice_weather,
        shapes=pvdeg.pysam.INSPIRE_GEOSPATIAL_TEMPLATE_SHAPES,
        add_dims={'distance':10}
    )

    # force align index, this is not ideal because it could obscure timezone errors
    # both are in UTC and off by a multiple of years so this is fine
    slice_template = slice_template.assign_coords({"time": slice_weather.time})

    partial_res = pvdeg.geospatial.analysis(
        weather_ds=slice_weather,
        meta_df=slice_meta,
        func=pvdeg.pysam.inspire_ground_irradiance,
        template=slice_template,
        config_files={
            'pv' : f"{conf_dir}/{conf}/{conf}_pvsamv1.json"
        }
    )


    gid_start = sub_gids[0]
    gid_end = sub_gids[-1]

    # ideally we write everything to the same zarr store but need to determine chunking and dimensions
    fname = Path(f"{target_dir}/{conf}/{gid_start}-{gid_end}.nc")
    Path(fname).parent.mkdir(parents=True, exist_ok=True)

    partial_res.to_netcdf(fname)
    logger.info(f"saved to file | {fname.resolve()}")

    # make sure we get rid of references to lazy objects and files at the end of each task (helps gc)
    del geo_weather
    del geo_meta
    del slice_weather
    del slice_meta
    del partial_res
    del template
    gc.collect()

def run_state(
    state: str,
    target_dir: Path,
    conf_dir: Path,
    dask_client,
    dask_nworkers:int,
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
   
    init_weather, init_meta, _chunk_size = load_weather(local_test_paths=local_test_paths, state=state)
    gids = init_meta.index.values

    logger.info(f"PREVIEW: \n{init_weather}\n")
    logger.info(f"PREVIEW: \n{init_meta.iloc[0:5]}\n")

    del init_weather
    del _chunk_size
    gc.collect()

    results = [] # delayed objects list

    # chunk size of 10, use 16 workers for this default (run as substeps so it can be handled in memory)
    # build the delayed graph
    # step_size = 640  # works but bad WHEN RUNNING 2 tasks at a time in the loop to limit parallelisim and memory usage
    step_size = 320 # 8 workers, chunk size of 40
    for conf in confs:

        # initialize full zarr store
        # full_output.to_zarr(config_zarr_path, compute=False, mode='w')

        # create delayed objects for each slice
        for i in range(0, len(gids), step_size):
            front = i
            back = min(i + step_size, len(init_meta) - 1)

            sub_gids = gids[front:back]

            results.append(
                process_slice(
                    conf=conf,
                    conf_dir=conf_dir,
                    target_dir=target_dir,
                    sub_gids=sub_gids,
                    local_test_paths=local_test_paths,
                    state=state
                    # geo_weather=geo_weather,
                    # geo_meta=geo_meta,
                    # front=front,
                    # back=back,
                )
            )

    # 40 locations takes 13 GB

    writes = [combine_many_nc_to_zarr(conf=conf, target_dir=target_dir) for conf in confs]

    logger.info(f"PREVIEW: delayed compute chunks : #{len(results)} ")
    logger.info(f"PREVIEW: delayed write tasks    : #{len(writes)} ")

    batch_size = int(dask_nworkers / 4) # send one chunk to each dask worker at a time
    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]

        with ProgressBar():
            compute(*batch)

        # force garbage collect on workers
        dask_client.run(gc.collect)

    with ProgressBar():
        compute(*writes)
    
    return
