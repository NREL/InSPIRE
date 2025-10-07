import json

PATH = "/projects/inspire/PySAM-MAPS/v1/model-outs/completeness.json"

with open(PATH, 'r') as fd:
    completeness = json.load(fd)

fields = [
    "albedo",
    "dhi",
    "dni",
    "ghi",
    "ground_irradiance",
    "relative_humidity",
    "subarray1_celltemp",
    "subarray1_dc_gross",
    "subarray1_poa_front",
    "subarray1_poa_rear",
    "temp_air",
    "wind_direction",
    "wind_speed",
]
confs = [
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
]

for state in completeness:
    for conf in confs:
        stats = completeness[state][conf]

        for field in fields:
            if stats[field] != 0:
                print(state, conf, "found non-zero missing gids")
