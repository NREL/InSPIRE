import glob
import numpy as np
import xarray as xr
import dask.array as da
import os
from pathlib import Path

from inspire_agrivolt import logger

def check_datasets_coords_dims(files: list[str]) -> None:
    from glob import glob
    import xarray as xr

    allowed = {"time", "distance", "gid"}

    for fp in files:
        try:
            ds = xr.open_dataset(str(fp), engine="netcdf4", decode_cf=False, mask_and_scale=False)
            coords = set(ds.coords.keys())
            dims = set(ds.sizes.keys())

            all_present = coords.union(dims)
            extra = all_present - allowed

            if extra:
                raise ValueError(f"{fp} contains unexpected dimensions or coordinates: {extra}")

            logger.debug(f"good: {fp} | coords: {sorted(coords)} | dims: {sorted(dims)}")

        except Exception as e:
            raise ValueError(f"bad: {fp} | FAILED: {e}") from e



def check_completeness(outputs_dir: str, state: str, conf: str) -> dict:
    """
    Check completeness of outputs for a state and config.
    Validates both CSV and NetCDF outputs for missing or extra GIDs.

    tree of ``outputs_dir``
    /projects/inspire/PySAM-Maps/Full-Outputs/
    - colorado
        - 01
        - 02
        - ..
        - 10
    - north carolina
        - 01
        - 02
        - ..
        - 10

    Returns:
        dict: Status summary with lists of missing/extra gids, and gid consistency info.
    """
    import pvdeg
    import pandas as pd
    import xarray as xr
    from pathlib import Path

    outputs_path = Path(outputs_dir)
    target_path = outputs_path / state / conf

    files = sorted(target_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {target_path}")

    meta_list = [pd.read_csv(file, index_col=0) for file in files]
    out_meta_df = pd.concat(meta_list)

    # check dims/coords before reading NetCDF
    check_datasets_coords_dims(files=sorted(target_path.glob("*.nc")))
    out_model_ds = xr.open_mfdataset(str(target_path / "*.nc"))

    # Get weather meta for full expected set of GIDs
    weather_db = "NSRDB"
    weather_arg = {
        "satellite": "Americas",
        "names": "TMY",
        "NREL_HPC": True,
        "attributes": pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
    }

    geo_weather, geo_meta = pvdeg.weather.get(
        weather_db, geospatial=True, **weather_arg
    )

    # All GIDs that should be present for the state
    sub_meta = geo_meta[geo_meta["state"] == state.title()]
    expected_gids = set(sub_meta.index.values)

    found_gids = set(out_meta_df.index.values)

    missing = list(expected_gids - found_gids)
    extra = list(found_gids - expected_gids)

    # GID consistency check between CSV and NetCDF outputs
    try:
        ds_gids = set(out_model_ds.gid.values)
    except AttributeError:
        raise ValueError("NetCDF dataset does not contain 'gid' coordinate.")

    # CSV index vs NetCDF gids
    csv_vs_netcdf_diff = found_gids.symmetric_difference(ds_gids)
    gid_consistent = len(csv_vs_netcdf_diff) == 0

    status = (
        "complete" if not missing and not extra and gid_consistent else
        "inconsistent" if not gid_consistent else
        "missing" if missing else
        "extra"
    )

    return {
        "status": status,
        "missing_gids": missing,
        "extra_gids": extra,
        "total_expected": len(expected_gids),
        "total_found_csv": len(found_gids),
        "total_found_netcdf": len(ds_gids),
        "csv_netcdf_gid_match": gid_consistent,
        "gid_mismatch_list": list(csv_vs_netcdf_diff)
    }

def generate_missing_gids_file(completeness: dict, output_path: str = None, filename: str = "missing_gids.txt") -> str:
    """
    takes dict created by check_completeness
    """

    if output_path is None:
        output_path = os.getcwd()

    os.makedirs(output_path, exist_ok=True) 

    missing_gids = completeness["missing_gids"]    
    missing_gids = [str(entry) for entry in missing_gids]
    data = " ".join(missing_gids)

    file_path = os.path.join(output_path, filename)

    with open(file_path, "w") as fp:
        fp.write(data)

    return file_path

def _validate_netcdf_files(nc_files):
    """
    Check if netcdf files can be opened. 
    
    Sometimes they get corrupted or written wrong. Function checks for this 
    to avoid opening or converting bad NC files into Zarr.
    """
    good = []
    bad = []
    for f in nc_files:
        try:
            xr.open_dataset(f).close()
            good.append(f)
        except Exception as e:
            print(f"BAD FILE: {f}")
            print(e)
            bad.append(f)
    return good, bad

def merge_original_fill_data_to_zarr(state: str, conf: str, full_outputs_dir: Path, fill_outputs_dir: Path) -> None:
    """
    Grabs state original output files, grab fill files, merge them and store to zarr.

    Parameters
    ----------
    state: str
        state name, lowercase
    conf: str
        conf name with leading zeros, ex: 04, 08, 09
    full_outputs_dir: pathlib.Path
        path to full outputs directory with state subdirectories

    example (assuming the following paths exist)
    .../path/to/colorado/
    .../path/to/colorado-fill/

    run with the following to merge for config 04
    >>> mix_state_from_original_filled "colorado" "04" .../path/to/
    """
    import pvdeg
    print(f"starting {state} {conf}")

    zarr_path = full_outputs_dir / state / f"{conf}.zarr"
    if zarr_path.exists():
        raise Exception(f"Zarr path already exists at {zarr_path}")

    original_files = list(full_outputs_dir.glob(f"{state}/{conf}/*.nc"))
    original_files, bad_original_files = _validate_netcdf_files(original_files)

    if bad_original_files:
        print("skipping bad original nc files:", bad_original_files)
    
    fill_files = list(fill_outputs_dir.glob(f"{state}/{conf}/*.nc"))
    fill_files, bad_fill_files = _validate_netcdf_files(fill_files)

    if bad_fill_files:
        print("skipping bad fill nc files:", bad_fill_files)

    # load good files, combine into single dataset
    fill_datasets = [xr.open_dataset(f).sortby('gid') for f in sorted(fill_files)]

    if fill_datasets:
        fill_combined = xr.concat(fill_datasets, dim='gid')
        fill_combined = fill_combined.sortby('gid')
    
    # load original incomplete dataset
    original_dataset = xr.open_mfdataset(original_files)

    # fill missing values in original dataset
    # we should try merging or concatenating some other way
    # filled_dataset = original_dataset.combine_first(fill_combined)
    if fill_datasets:
        merged = xr.concat([original_dataset, fill_combined], dim="gid").sortby("gid") # combine datasets (may have duplicates), sortby gid for grouping
    else:
        merged=original_dataset.sortby("gid")
    _, unique_indices = np.unique(merged["gid"].values, return_index=True) # take first entry for each gid
    filled_dataset = merged.isel(gid=unique_indices)

    # load nsrdb metadata
    weather_db = "NSRDB"
    weather_arg = {
        "satellite": "Americas",
        "names": "TMY",
        "NREL_HPC": True,
        "attributes": pvdeg.pysam.INSPIRE_NSRDB_ATTRIBUTES,
    }
    
    geo_weather, geo_meta = pvdeg.weather.get(
        weather_db, geospatial=True, **weather_arg)

    # get state metadata
    state_meta = geo_meta[geo_meta["state"] == f"{state.title()}"]

    # check if any gids are outside of the state, if so remove them, this seems to have been an issue in the past
    # while probably not required, we want to make sure that this is not a slient issue
    valid_gids = state_meta.index.values
    mask = np.isin(filled_dataset.gid.values, valid_gids) # mask values to keep
    filtered_dataset = filled_dataset.isel(gid=mask).chunk({"gid":100})

    filtered_dataset.to_zarr(full_outputs_dir / state / f"{conf}.zarr")

    print(f"ending {state} {conf}")