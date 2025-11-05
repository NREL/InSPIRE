"""
Script for parallelizing uploads from Kestrel to S3.
"""

import os
import sys
import logging
from collections.abc import Iterable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig

# -----------------------------------------------------------
DRY_RUN = False

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

OEDI_STAGING_BUCKET = "oedi-data-drop"
HTTP_MAX_POOL_CONNECTIONS = 64
MULTIPROCESSING_MAX_WORKERS = 32

MULTIPART_MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
MULTIPART_THRESHOLD = 32 * 1024 * 1024  # 16 MiB
MULTIPART_CHUNK = 64 * 1024 * 1024  # 32 MiB
# -----------------------------------------------------------

logger = logging.getLogger("uploader")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def iter_local_files(root: Path) -> Iterable[Path]:
    """
    Create an iterable for local files.
    """
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def build_key(root: Path, file: Path, inspire_prefix: str) -> str:
    rel = file.relative_to(root).as_posix()
    return f"{inspire_prefix}{rel}"


def upload_one(
    s3_client, file: Path, bucket: str, key: str, transfer_cfg: TransferConfig
) -> tuple[str, int]:
    s3_client.upload_file(
        Filename=str(file), Bucket=bucket, Key=key, Config=transfer_cfg, ExtraArgs={}
    )
    return key, file.stat().st_size


def main() -> None:
    args = sys.argv
    if len(args) != 2:
        raise RuntimeError("expected usage <python upload.py zarr-name>")

    zarr = args[1]
    print(zarr)

    INSPIRE_PREFIX = f"inspire/{zarr}/"
    LOCAL_ROOT = Path(f"/projects/inspire/PySAM-MAPS/v1.1/final/{zarr}")

    if not LOCAL_ROOT.exists():
        raise NotADirectoryError(f"Dir {str(LOCAL_ROOT)} not found")

    if AWS_ACCESS_KEY_ID is None or AWS_SECRET_ACCESS_KEY is None:
        raise RuntimeError(
            "Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY env vars"
        )

    session_kwargs = {"region_name": "us-west-2"}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        session_kwargs.update(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    s3_client = boto3.client(
        "s3",
        config=Config(
            max_pool_connections=HTTP_MAX_POOL_CONNECTIONS,
            retries={"mode": "adaptive", "max_attempts": 10},
        ),
        **session_kwargs,
    )

    # per file config
    transfer_cfg = TransferConfig(
        multipart_threshold=MULTIPART_THRESHOLD,
        multipart_chunksize=MULTIPART_CHUNK,
        max_concurrency=MULTIPART_MAX_WORKERS,
        use_threads=True,
    )

    to_upload: list[tuple[Path, str, int]] = []
    total_bytes = 0

    for f in iter_local_files(LOCAL_ROOT):
        key = build_key(LOCAL_ROOT, f, inspire_prefix=INSPIRE_PREFIX)
        size = f.stat().st_size

        to_upload.append((f, key, size))
        total_bytes += size

    logger.info(
        f"Planned {len(to_upload)} uploads ({total_bytes / 1024 / 1024 / 1024:.2f} GiB)." 
        f"Starting with up to {MULTIPROCESSING_MAX_WORKERS} workers..."
    )
    
    if DRY_RUN:
        exit()

    errors = 0
    futures = []
    with ThreadPoolExecutor(max_workers=MULTIPROCESSING_MAX_WORKERS) as pool:
        for f, key, _ in to_upload:
            futures.append(
                pool.submit(
                    upload_one, s3_client, f, OEDI_STAGING_BUCKET, key, transfer_cfg
                )
            )

        for i, fut in enumerate(as_completed(futures), 1):
            try:
                key, sz = fut.result()
                if i % 100 == 0 or i == len(futures):
                    logger.info(
                        "Progress: %d/%d (last: %s, %d bytes)", i, len(futures), key, sz
                    )
            except ClientError as e:
                errors += 1
                logger.error("Upload failed: %s", e, exc_info=False)

    logger.info("Done. Successful: %d  Failed: %d", len(futures) - errors, errors)


if __name__ == "__main__":
    main()
