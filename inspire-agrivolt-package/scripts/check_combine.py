from pathlib import Path
import xarray as xr
from dask.distributed import Client, LocalCluster
import argparse

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("final-dir", default=str, help='absolute path to directory containing combined model-outs and postprocessing results')
    args = parser.parse_args()

    final_dir = Path(args.final_dir)

    print("running check_combine.py")
    client = Client(LocalCluster(n_workers=31))

    for conf in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]:
        print(f"running conf: {conf}")
        conf_zarr_path = final_dir / f"{conf}.zarr"

        if not conf_zarr_path.exists():
            raise FileNotFoundError(f"path does not exist: {str(conf_zarr_path)}")

        conf_zarr = xr.open_zarr(str(conf_zarr_path))

        for var in conf_zarr.data_vars:
            print(conf, var, conf_zarr[var].isnull().values.any())

if __name__ == "__main__":
    print("running check_combine.py")

    main()