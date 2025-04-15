import argparse
from getpass import getuser
from pathlib import Path
from pvdeg.utilities import nrel_kestrel_check

from inspire_agrivolt import logger

def ground_irradiance():
    log_directory = f"/tmp/{getuser()}/agrivoltaics-irradiance-logs"

    parser = argparse.ArgumentParser(
        description="Run Agrivoltaics Irradiance PySAM simulation for a U.S. state"
    )

    parser.add_argument(
        "state",
        type=str,
        help="State name to run analysis on as it appears in NSRDB meta data, i.e. 'Colorado'. Only used when pulling NSRDB data",
    )
    parser.add_argument("target_dir", type=str, help="path to base output directory")
    parser.add_argument("conf_dir", type=str, help="path to base configs directory")

    parser.add_argument(
        "--confs",
        nargs="+",
        default=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"],
        help="List of config names (default: 01 through 10)",
    )

    parser.add_argument(
        "--partition",
        type=str,
        help="slurm partition to use when scheduling",
        default="shared",
    )
    parser.add_argument(
        "--account",
        type=str,
        help="nrel hpc account with allocation",
        default="inspire",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="number of dask workers (max 64 on kestrel shared partition)",
        default=16,
    )
    parser.add_argument(
        "--walltime",
        type=str,
        help="max length of job from start to finish",
        default="02:00:00",
    )

    parser.add_argument(
        "--log_dir", type=str, help="location of dask log files", default=log_directory
    )
    parser.add_argument("--port", type=int, help="dask dashboard port", default=12345)

    parser.add_argument(
        "--local-weather", type=Path, help="Path to local weather NetCDF file"
    )
    parser.add_argument("--local-meta", type=Path, help="Path to local meta CSV file")

    args = parser.parse_args()

    from inspire_agrivolt import irradiance_sam
    from dask.distributed import LocalCluster, Client
    from dask_jobqueue import SLURMCluster


    # convert to dict if both paths are provided
    local_test_paths = None

    if args.local_weather and args.local_meta:
        print(
            "local files provided, skipping NSRDB, using dask with 8 workers, dask logs not saved"
        )
        local_test_paths = {"weather": args.local_weather, "meta": args.local_meta}

        workers = 8

        logger.info("starting localcluster on local machine")
        cluster = LocalCluster(
            n_workers=workers,
            processes=True,
        )

        client = Client(cluster)

        import webbrowser
        webbrowser.open(client.dashboard_link)

    elif (not args.local_weather and args.local_meta) or (
        args.local_weather and not args.local_meta
    ):
        raise ValueError("must provide both local_weather and local_meta or neither")

    else:
        nrel_kestrel_check() # will only pass if we are on kestrel
        logger.info("starting slurmcluster on kestrel")

        # threads is cores / processes according to documentation
        # https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html
        cluster = SLURMCluster(
            queue=args.partition,
            account=args.account,
            cores=args.workers,         # n cores per worker
            processes=args.workers,     # n independent processes per worker
            memory="120GB",
            log_directory=args.log_dir,
            walltime=args.walltime,
            scheduler_options={
                "dashboard_address": f":{args.port}"
            },
        )
        cluster.scale(args.workers)
        client = Client(cluster)
        print(f"dask dashboard link (must port forward if on HPC): {client.dashboard_link}")


    # logger.info(f"running ground irradiance calculation using SAM for {args.state}")
    logger.info(f"RUNNING GEOSPATIAL GROUND IRRADIANCE CALCULATION USING SAM AND PVDEG FOR: {args.state}")

    # run function
    irradiance_sam.run_state(
        state=args.state.title(),
        target_dir=Path(args.target_dir),
        conf_dir=Path(args.conf_dir),
        confs=args.confs,
        local_test_paths=local_test_paths,
        dask_client=client,
        dask_nworkers=args.workers,
    )


    client.close()
