import glob
import numpy as np
import xarray as xr
import dask.array as da

# files = glob.glob("/projects/inspire/PySAM-MAPS/Full-Outputs/Colorado/01/*.nc")
# zarr_path = "/projects/inspire/PySAM-MAPS/Full-Outputs/Colorado/01/merged.zarr"

def pysam_output_netcdf_to_zarr(files: list[str], zarr_path) -> None:
    """
    merge many netcdf (.nc) outputs from ``ground_irradiance`` to a single zarr store.

    *this function is not ideal, ideally we would write to the zarr store at the time of calculation for each chunk in ``ground_irradiance`` rather than intermediately saving outputs to .nc*

    Parameters
    -----------
    files: list[str]
        list of filenames to merge (must all have same dimensions)
    zarr_path: str
        path to zarr store (creates store at this path)

    Returns
    -------
    None
    """

    lat_set, lon_set = set(), set()
    for file in files:
        with xr.open_dataset(file, engine="netcdf4", chunks={}) as ds:
            lat_set.update(ds.latitude.values.tolist()) # add all elements in iterables to sets
            lon_set.update(ds.longitude.values.tolist())
            
    unique_lat = np.sort(np.fromiter(lat_set, dtype=float))
    unique_lon = np.sort(np.fromiter(lon_set, dtype=float))

    example_ds = xr.open_dataset(files[0], engine="netcdf4")

    chunks = {"latitude": 40, "longitude":10}
    sizes={"latitude":len(unique_lat), "longitude":len(unique_lon), "time":8760, "distance":10}

    data_vars = {}
    for data_var in example_ds.data_vars:
        
        dims = example_ds[data_var].dims
        
        shape = tuple(sizes[d] for d in dims)
        chunk_shape = tuple(chunks.get(d, sizes[d]) for d in dims)
        dtype = example_ds[data_var].dtype

        data_vars[data_var] = (dims, da.full(shape, np.nan, dtype=dtype, chunks=chunk_shape))
        
    template = xr.Dataset(
        coords={
            "latitude":unique_lat,
            "longitude":unique_lon,
            "time":example_ds.time,
            "distance":np.arange(10)
        },
        data_vars=data_vars
    )

    template.to_zarr(zarr_path, mode='w')

    merged = xr.open_zarr(zarr_path, consolidated=False)

    for file in files:
        ds = xr.open_dataset(file, chunks={}, engine="netcdf4")
        
        lat_inds = np.searchsorted(unique_lat, ds.latitude.values)
        lon_inds = np.searchsorted(unique_lon, ds.longitude.values)

        region = {
            "latitude": slice(lat_inds[0], lat_inds[-1] + 1),
            "longitude": slice(lon_inds[0], lon_inds[-1] + 1),
            "time":slice(0,8760),
            "distance":slice(0,10)
        }
        
        existing_block = merged.isel(latitude=region["latitude"], longitude=region["longitude"])

        ref_var = "annual_poa"
        mask = np.isnan(existing_block[ref_var]) & ~np.isnan(ds[ref_var])
        
        patch_data = {}
        for var in ds.data_vars:
            patch_data[var] = (
                ds[var].dims, 
                np.where(mask, ds[var].values, existing_block[var].values)
            )
                        
        patch_coords = {
            "time": merged.time.values,
            "distance": merged.distance.values,
            "latitude": merged.latitude.values[region["latitude"]],
            "longitude": merged.longitude.values[region["longitude"]],
        }
                        
        patch = xr.Dataset(
            data_vars=patch_data,
            coords=patch_coords
        )
        
        patch.to_zarr(zarr_path, region=region, mode="r+")