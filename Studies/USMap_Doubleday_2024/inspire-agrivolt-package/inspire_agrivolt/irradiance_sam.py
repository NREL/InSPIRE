import pvdeg
from pathlib import Path
from dask.diagnostics import ProgressBar
from dask import delayed, compute
import gc
import xarray as xr
import pandas as pd
import time
import numpy as np
import traceback

from inspire_agrivolt import logger

WEATHER_DB = "NSRDB"
WEATHER_ARG = {
    "satellite": "Americas",
    "names": "TMY",
    "NREL_HPC": True,
    "attributes": pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
}

def get_local_weather(local_test_paths: dict) -> tuple[xr.Dataset, pd.DataFrame, int]:
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
        logger.info("using placeholder albedo, filling 0.2 for all timesteps")
        geo_weather = geo_weather.assign(albedo=geo_weather["temp_air"] * 0 + 0.2) 

    logger.info(f"{len(geo_meta)} location entries in local data")

    # geo_weather['time'] = pd.date_range(start="2001-01-01 00:30:00", periods=8760, freq='1h')
    chunk_size = 10

    return geo_weather, geo_meta, chunk_size

def _load_full_nsrdb_kestrel() -> tuple[xr.Dataset, pd.DataFrame]:
    logger.info("LOADING: weather dataset from NSRDB on kestrel")
    start = time.time()

    geo_weather, geo_meta = pvdeg.weather.get(
        WEATHER_DB, geospatial=True, **WEATHER_ARG
    )

    end = time.time()
    logger.info(f"LOADED: dataset in: {end - start} s")

    return geo_weather, geo_meta


def load_weather_state(
    local_test_paths: dict, 
    state: str,
) -> tuple[xr.Dataset, pd.DataFrame, int]:
    if local_test_paths is not None:
        logger.info("LOADING: weather dataset from local files")
        geo_weather, geo_meta, chunk_size = get_local_weather(local_test_paths=local_test_paths)

    else:
        geo_weather, geo_meta = _load_full_nsrdb_kestrel()

        # downselect NSRDB to requested gids
        logger.info(f"DOWNSELECTING: keeping points where 'state'== '{state}'")
        start = time.time()
        geo_meta = geo_meta[geo_meta["country"] == "United States"]
        geo_meta = geo_meta[geo_meta["state"] == state]
        end = time.time()
        logger.info(f"DOWNSELECTING: took {end - start} seconds")
        
        chunk_size = 40

        ############### force downsample for testing
        geo_meta, gids_sub = pvdeg.utilities.gid_downsampling(meta=geo_meta, n=15)
        ###############

    geo_weather = geo_weather.sel(gid=geo_meta.index).chunk({"gid": chunk_size})

    return geo_weather, geo_meta, chunk_size


def load_weather_gids(
    local_test_paths: dict, 
    gids: np.ndarray,
) -> tuple[xr.Dataset, pd.DataFrame, int]:
    if local_test_paths is not None:
        logger.info("LOADING: weather dataset from local files")
        geo_weather, geo_meta, chunk_size = get_local_weather(local_test_paths=local_test_paths)

    else:
        geo_weather, geo_meta = _load_full_nsrdb_kestrel()

    chunk_size = 40

    logger.info(f"DOWNSELECTING: keeping points from provided #{len(gids)} gids")
    start = time.time()

    geo_weather = geo_weather.sel(gid=gids).chunk({"gid": chunk_size})
    geo_meta = geo_meta.loc[gids]

    end = time.time()
    logger.info(f"DOWNSELECTING: took {end - start} seconds")

    return geo_weather, geo_meta, chunk_size


@delayed
def process_slice(
    conf: str,
    conf_dir: str,
    target_dir: str,
    sub_gids: np.ndarray,
    local_test_paths: dict=None,
) -> None:
    slice_weather, slice_meta, chunk_size = load_weather_gids(
        gids=sub_gids,                      # subset of gids from state, determined in run_state()
        local_test_paths=local_test_paths,  # usually None
    )

    # update variable names to match convention
    slice_weather = pvdeg.weather.map_weather(slice_weather)

    slice_template = pvdeg.geospatial.output_template(
        ds_gids=slice_weather,
        shapes=pvdeg.pysam.INSPIRE_GEOSPATIAL_TEMPLATE_SHAPES,
        add_dims={'distance':10}
    )

    # force align index, this is not ideal because it could obscure timezone errors
    # both are in UTC and off by a multiple of years so this is fine
    tmy_time_index = pd.date_range("2001-01-01", periods=8760, freq='1h')
    slice_weather["time"] = tmy_time_index
    slice_template["time"] = tmy_time_index

    partial_res = pvdeg.geospatial.analysis(
        weather_ds=slice_weather,
        meta_df=slice_meta,
        func=pvdeg.pysam.inspire_ground_irradiance,
        template=slice_template,
        config_files={
            'pv' : f"{conf_dir}/{conf}/{conf}_pvsamv1.json"
        }
    )

    # all gids in range between gid_start and gids_end ARE NOT GUARANTEED CONTAINED THE RESULT
    gid_start = sub_gids[0]
    gid_end = sub_gids[-1]

    # ideally we write everything to the same zarr store but need to determine chunking and dimensions
    try:
        fname = f"{target_dir}/{conf}/{gid_start}-{gid_end}.nc"
        file_path = Path(fname)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        partial_res.to_netcdf(fname)
        logger.info(f"saved to file | {file_path.resolve()}")
    except Exception as e:
        task_info = f"Saving partial result to NetCDF file: {fname}"
        error_msg = f"Error during task: {task_info}\nOriginal error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        raise Exception(error_msg) from e

    # make sure we get rid of references to lazy objects and files at the end of each task (helps gc)
    # we shouldn't have to do this but dask was struggling with memory blow up
    del slice_weather
    del slice_meta
    del partial_res
    del slice_template
    gc.collect()

    return

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
    gids: np.ndarray = None,
) -> None:
   
    # if gids are not provided then we gather them from the NSRDB to determine number of iterations
    if gids is None: 
        init_weather, init_meta, _chunk_size = load_weather_state(
            local_test_paths=local_test_paths, 
            state=state, 
        )

        # get gids for designated state
        gids = init_meta.index.values

        logger.info(f"PREVIEW: \n{init_weather}\n")
        logger.info(f"PREVIEW: \n{init_meta.iloc[0:5]}\n")

        del init_weather
        del init_meta
        del _chunk_size
        gc.collect()

    results = [] # delayed objects list

    # chunk size of 10, use 16 workers for this default (run as substeps so it can be handled in memory)
    # build the delayed graph
    # step_size = 640  # works but bad WHEN RUNNING 2 tasks at a time in the loop to limit parallelisim and memory usage
    step_size = 320 # 8 workers, chunk size of 40
    for conf in confs:

        # create delayed objects for each slice
        for i in range(0, len(gids), step_size):
            front = i
            back = min(i + step_size, len(gids))

            # we can build a futures list if we scatter this outside of the array then pass the futures and iterate again
            sub_gids = gids[front:back]

            results.append(
                process_slice(
                    conf=conf,
                    conf_dir=conf_dir,
                    target_dir=target_dir,
                    sub_gids=sub_gids,                      # only run simulation on these gids (provided, or determined at top of function)
                    local_test_paths=local_test_paths,      # usually None
                )
            )

    logger.info(f"PREVIEW: delayed compute chunks : #{len(results)} ")

    batch_size = max(1, dask_nworkers // 4) # send one chunk to each dask worker at a time (keeps from blowing it up but i dont like this )
    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]

        with ProgressBar():
            compute(*batch)

        dask_client.run(gc.collect)
    
    return
