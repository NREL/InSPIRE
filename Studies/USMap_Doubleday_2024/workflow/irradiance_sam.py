import pvdeg
from pathlib import Path
from dask.diagnostics import ProgressBar
import os


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

    if local_test_paths is not None:
        import xarray as xr
        import pandas as pd

        # need to be tmy files
        geo_weather = xr.open_dataset(local_test_paths["weather"])
        geo_meta = pd.read_csv(local_test_paths["meta"], index_col=0)

        print(f"{len(geo_meta)} location entries in local data")

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

    pbar = ProgressBar()
    pbar.register()

    # run the analysis chunked into substeps here, don't want to run out of memory
    step_size = 640  # chunk size of 10, use 16 workers for this default
    for conf in confs:
        print(f"starting config | {conf}")

        for i in range(0, len(geo_meta), step_size):
            # ensure we dont go out of bounds
            front = i
            back = min(i + step_size, len(geo_meta) - 1)
            print(f"chunk index bounds: {front} -> {back}")

            slice_weather = geo_weather.isel(gid=slice(front, back))
            slice_meta = geo_meta.iloc[front:back]

            partial_res = pvdeg.geospatial.analysis(
                weather_ds=slice_weather,
                meta_df=slice_meta,
                func=pvdeg.standards.standoff,  # test function
                # extra = ...
                # {'pv' : conf_dir / conf / conf_pvsamv1.json}
            )

            fname = Path(f"{target_dir}/{conf}/{i}-{i+i-1}.nc")
            Path(fname).parent.mkdir(parents=True, exist_ok=True)

            partial_res.to_netcdf(fname)
            print(f"saved to file | {os.getcwd()}/{fname}")

    pbar.unregister()

    return
