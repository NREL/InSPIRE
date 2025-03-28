import pvdeg
from pathlib import Path
from dask.diagnostics import ProgressBar
import os
#import shutil
from tqdm import tqdm # progress bars

def run_state(
    state: str,
    target_dir: Path,
    conf_dir: Path,
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
    print(local_test_paths)

    # this is hacky and created for my local test files
    if local_test_paths is not None:
        import xarray as xr
        import pandas as pd

        # need to be tmy files
        geo_weather = xr.open_dataset(local_test_paths["weather"])
        geo_meta = pd.read_csv(local_test_paths["meta"], index_col=0)

        if "wind_direction" not in geo_weather.data_vars:
            print("using placeholder wind direction, filling zeros")
            geo_weather = geo_weather.assign(wind_direction=geo_weather["temp_air"] * 0)
        
        if "albedo" not in geo_weather.data_vars:
            print("using placeholder wind direction, filling 0.2 for all timesteps")
            geo_weather = geo_weather.assign(albedo=geo_weather["temp_air"] * 0 + 0.2) 

        print(f"{len(geo_meta)} location entries in local data")

        # coerce time to be same as template, year doesnt matter because it is tmy
        # could move this to the analysis
        geo_weather['time'] = pd.date_range(start="2001-01-01 00:30:00", periods=8760, freq='1h')

    else:
        weather_db = "NSRDB"
        weather_arg = {
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

        geo_weather, geo_meta = pvdeg.weather.get(
            weather_db, geospatial=True, **weather_arg
        )

        # only need to do this if coming from the NSRDB
        geo_meta = geo_meta[geo_meta["country"] == "United States"]
        geo_meta = geo_meta[geo_meta["state"] == state]

    # downselect and chunk
    geo_weather = geo_weather.sel(gid=geo_meta.index)
    geo_weather = geo_weather.chunk({"gid": 40})

    # make sure we are in the middle of the hour for position calculations
    # this doesn't matter for pysam


    # run the analysis chunked into substeps here, don't want to run out of memory
    # chunk size of 10, use 16 workers for this default
    step_size = 640  
    for conf in tqdm(confs):
        print(f"starting config | {conf}")

        for i in tqdm(range(0, len(geo_meta), step_size)):
            # ensure we dont go out of bounds
            front = i
            back = min(i + step_size, len(geo_meta) - 1)
            print(f"chunk index bounds: {front} -> {back}")

            slice_weather = geo_weather.isel(gid=slice(front, back))
            slice_meta = geo_meta.iloc[front:back]

            template = pvdeg.geospatial.output_template(
                ds_gids=slice_weather,
                shapes=pvdeg.pysam.INSPIRE_GEOSPATIAL_TEMPLATE_SHAPES,
                add_dims={'distance':10}
            )

            # we may want to force slice_weather.index to be the same as template.index
            # they are tmy and in the middle of the hour already but could have different years
            # tmy so year does not matter, it is just there as a placeholder

            partial_res = pvdeg.geospatial.analysis(
                weather_ds=slice_weather,
                meta_df=slice_meta,

                func=pvdeg.pysam.inspire_ground_irradiance,
                template=template,
                config_files={
                    'pv' : f"{conf_dir}/{conf}/{conf}_pvsamv1.json"
                }
            )

            fname = Path(f"{target_dir}/{conf}/{i}-{i+i-1}.nc")
            Path(fname).parent.mkdir(parents=True, exist_ok=True)

            partial_res.to_netcdf(fname)
            print(f"saved to file | {os.getcwd()}/{fname}")

        print(f"merging files for {conf}")
        combined_ds = xr.open_mfdataset(f"{target_dir}/{conf}/*.nc")
        
        combined_ds.to_netcdf(f"{target_dir}/{conf}-pvdeg-pysam.nc")

    return
