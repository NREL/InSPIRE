from importlib.metadata import version
import sys
import logging

logger = logging.getLogger("inspire_agrivolt")
#logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from . import main
from . import irradiance_sam

from .file_operations import merge_pysam_out_nc_to_zarr
from .verify import verify_dataset_gids

__version__ = version("inspire_agrivolt")
