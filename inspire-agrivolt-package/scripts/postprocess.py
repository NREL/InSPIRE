import inspire_agrivolt
from dask.distributed import LocalCluster
from pathlib import Path

STATES = [
    "alabama",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new_hampshire",
    "new_jersey",
    "new_mexico",
    "new_york",
    "north_carolina",
    "north_dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode_island",
    "south_carolina",
    "south_dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west_virginia",
    "wisconsin",
    "wyoming",
]

CONFS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
VERSION = "v1.1"

def check_all_state_confs_exist():
    for state in STATES:
        for conf in CONFS:
            print(state, conf)
            path = Path(f"/projects/inspire/PySAM-MAPS/{VERSION}/model-outs/{state}/{conf}")

            if not path.exists():
                raise ValueError(f"{state}/{conf} doesn't exist")

def run_all():
    for state in STATES:
        for conf in CONFS:
            print(state, conf)
            path = Path(f"/projects/inspire/PySAM-MAPS/{VERSION}/model-outs/{state}/{conf}")

            model_outs_paths = list(
                path.glob("*.zarr")
            )

            POSTPROCESS_OUTS_ZARR = Path(
                f"/projects/inspire/PySAM-MAPS/{VERSION}/postprocess/{state}/{conf}.zarr"
            )

            inspire_agrivolt.beds_postprocessing.postprocessing(
                scenario=conf,
                input_zarr_paths=model_outs_paths,
                output_zarr_path=POSTPROCESS_OUTS_ZARR,
            )

if __name__ == "__main__":
    check_all_state_confs_exist()

    cluster = LocalCluster(
        n_workers=16, 
        threads_per_worker=1
    )
    client = cluster.get_client()

    print(client.dashboard_link)
    run_all()
