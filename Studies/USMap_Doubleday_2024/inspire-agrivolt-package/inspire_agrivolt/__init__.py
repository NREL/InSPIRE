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

from .irradiance_sam import load_weather
from .file_operations import pysam_output_netcdf_to_zarr

__version__ = version("inspire_agrivolt")
