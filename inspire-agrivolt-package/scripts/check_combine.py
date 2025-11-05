from pathlib import Path
import xarray as xr
from dask.distributed import Client, LocalCluster

def main():
    print("running main")
    client = Client(LocalCluster(n_workers=31))

    for conf in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
        print(conf)

        final_dir = Path("/projects/inspire/PySAM-MAPS/v1.1/final/")
        conf_zarr_path = final_dir / f"{conf}.zarr"

        conf_zarr = xr.open_zarr(str(conf_zarr_path))

        for var in conf_zarr.data_vars:
            print(conf, var, conf_zarr[var].isnull().values.any())


if __name__ == "__main__":
    print("in python script")

    main()