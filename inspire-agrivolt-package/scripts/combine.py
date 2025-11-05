from pathlib import Path
import numpy as np
import xarray as xr

from dask.distributed import LocalCluster, Client

from inspire_agrivolt.beds_postprocessing import ground_irradiance_distances

MODEL_OUTS_DIR = Path("/projects/inspire/PySAM-MAPS/v1.1/model-outs/")
POSTPROCESS_OUTS_DIR = Path("/projects/inspire/PySAM-MAPS/v1.1/postprocess/")

def load_model_outs_zarrs(confs: list[str]) -> dict[str, list[xr.Dataset]]: 
    """
    Find and load all model outputs results zarrs.
    """
    # FIND all MODEL_OUTS paths by config
    model_outs_all_conf_zarrs_paths = {}
    for conf in confs:
        conf_zarrs_paths = model_outs_all_conf_zarrs_paths.get(conf, [])
        model_outs_all_conf_zarrs_paths[conf] = conf_zarrs_paths

        for dir in MODEL_OUTS_DIR.iterdir():
            for conf_dir in dir.glob(conf):
                model_outs_all_conf_zarrs_paths[conf] += list(conf_dir.glob("*.zarr"))
        print(f"found {len(model_outs_all_conf_zarrs_paths[conf])} zarrs for model outs conf {conf}")

    # LOAD all MODEL_OUTS zarrs by config
    model_outs_all_conf_zarrs = {}
    for conf in confs:
        print(f"loading model outs zarrs for conf {conf}")
        conf_zarrs = model_outs_all_conf_zarrs.get(conf, [])
        model_outs_all_conf_zarrs[conf] = conf_zarrs

        for path in model_outs_all_conf_zarrs_paths[conf]:
            model_outs_chunk = xr.open_zarr(path)
            model_outs_all_conf_zarrs[conf].append(model_outs_chunk)
        print(f"loaded {len(model_outs_all_conf_zarrs[conf])} MODEL OUTS zarrs to for config {conf}.")

    return model_outs_all_conf_zarrs

def load_postprocessing_zarrs(confs: list[str]) -> dict[str, list[xr.Dataset]]: 
    """
    Find and load all postprocessing results zarrs.
    """
    # find all postprocessing zarrs paths
    postprocess_all_conf_zarrs_paths = {}
    for conf in confs:
        conf_zarrs_paths = postprocess_all_conf_zarrs_paths.get(conf, [])
        postprocess_all_conf_zarrs_paths[conf] = conf_zarrs_paths

        for state_dir in POSTPROCESS_OUTS_DIR.iterdir():
            for conf_zarr in state_dir.glob(f"{conf}.zarr"):

                postprocess_all_conf_zarrs_paths[conf].append(conf_zarr)

    # load all postprocessing zarrs
    postprocess_all_conf_zarrs = {}
    for conf in confs:
        print(f"loading postprocessing zarrs for conf {conf}")
        conf_zarrs = postprocess_all_conf_zarrs.get(conf, [])
        postprocess_all_conf_zarrs[conf] = conf_zarrs

        for path in postprocess_all_conf_zarrs_paths[conf]:
            postprocess_state_conf = xr.open_zarr(path).drop_dims(10) # extra dim in dataset
            postprocess_all_conf_zarrs[conf].append(postprocess_state_conf)

        print(f"loaded {len(postprocess_all_conf_zarrs[conf])} POSTPROCESS zarrs to for config {conf}.")

    return postprocess_all_conf_zarrs

def main():
    print("running main")
    client = Client(
        LocalCluster(
            n_workers=31
        )
    )

    print(client.dashboard_link)

    confs = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    model_outs_all_conf_zarrs = load_model_outs_zarrs(confs)
    postprocess_all_conf_zarrs = load_postprocessing_zarrs(confs)

    for conf in confs:
        # combine model and postprocessing results
        model_res = xr.concat(model_outs_all_conf_zarrs[conf], dim="gid")
        postprocessing_res = xr.concat(postprocess_all_conf_zarrs[conf], dim="gid")
        combined = xr.merge([model_res, postprocessing_res])

        # calculate beds distances
        if "distances_m" not in combined.data_vars:
            print("distances_m not in combined result, calculating beds distances from pitch")
            distances_m_da = ground_irradiance_distances(combined)
        combined = xr.merge([combined, distances_m_da])

        # rename subarray1_* variables
        rename = {}
        for data_var in combined.data_vars:
            if data_var.startswith("subarray1_"):
                rename[data_var] = data_var.lstrip("subarray1_")
        
        # rename under_panel (misnamed in a postprocessing run)
        if "under_panel" in combined.data_vars:
            rename = rename | {"under_panel":"underpanel"}
        combined = combined.rename(rename)

        # convert to i32
        combined['distance'] = combined.distance.astype(np.int32)
        combined['gid'] = combined.gid.astype(np.int32)

        store_path = Path(f"/projects/inspire/PySAM-MAPS/v1.1/final/{conf}.zarr")
        if not store_path.exists():
            if 10 in combined.dims:
                combined = combined.drop_dims([10])
        
            # print rechunking
            combined = combined.chunk({"gid":40, "distance":-1, "time":-1})
            combined = combined.unify_chunks()
        
            # avoid rechunk during write
            encoding = {
                v: {"chunks": tuple(chunk_sizes) for chunk_sizes in combined[v].chunks}
                for v in combined.data_vars
            }

            # triggers computation, none has been done up to this point
            print("writing to path")
            combined.to_zarr(
                store_path, 
                compute=True,
                consolidated=True,
                encoding=encoding
            )

if __name__ == "__main__":
    main()