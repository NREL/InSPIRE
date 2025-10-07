import sys
from inspire_agrivolt.main import ground_irradiance

import sys
from inspire_agrivolt.main import ground_irradiance

# simulate arguments to the function as if it was called by its entrypoint/script
sys.argv = [
    "agrivolt_ground_irradiance", # entry point name being used in pyproject.toml
    "Colorado",
    "../Full-Outputs/Colorado/",
    "../SAM",
    "--confs", "01",
    "--port", "22118",
    "--workers", "32"
]

ground_irradiance()
