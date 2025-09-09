
import pandas as pd
import numpy as np
import xarray as xr
import dask.array as da

from collections.abc import Callable
from pathlib import Path


# TODO, calculate distances dynamically
# THESE WILL BECOME OUTDATED WHEN WE UPDATE THE CONFIGS (calculate dynamically)
# DISTANCES = {
#     "01": (
#         0.2502810919746036,
#         0.7508432759238107,
#         1.251405459873018,
#         1.751967643822225,
#         2.2525298277714323,
#         2.753092011720639,
#         3.2536541956698466,
#         3.7542163796190535,
#         4.2547785635682605,
#         4.755340747517468,
#     ),
#     "02": (
#         0.2502810919746036,
#         0.7508432759238107,
#         1.251405459873018,
#         1.751967643822225,
#         2.2525298277714323,
#         2.753092011720639,
#         3.2536541956698466,
#         3.7542163796190535,
#         4.2547785635682605,
#         4.755340747517468,
#     ),
#     "03": (
#         0.2502810919746036,
#         0.7508432759238107,
#         1.251405459873018,
#         1.751967643822225,
#         2.2525298277714323,
#         2.753092011720639,
#         3.2536541956698466,
#         3.7542163796190535,
#         4.2547785635682605,
#         4.755340747517468,
#     ),
#     "04": (
#         0.4004497471593658,
#         1.2013492414780975,
#         2.002248735796829,
#         2.8031482301155606,
#         3.6040477244342926,
#         4.404947218753024,
#         5.205846713071756,
#         6.006746207390488,
#         6.807645701709219,
#         7.608545196027951,
#     ),
#     "05": (
#         0.5561802043880081,
#         1.6685406131640241,
#         2.7809010219400405,
#         3.8932614307160565,
#         5.005621839492073,
#         6.117982248268089,
#         7.230342657044105,
#         8.342703065820121,
#         9.455063474596137,
#         10.567423883372154,
#     ),
#     "06": (
#         0.33370812263280486,
#         1.0011243678984145,
#         1.6685406131640244,
#         2.335956858429634,
#         3.003373103695244,
#         3.6707893489608536,
#         4.338205594226463,
#         5.005621839492073,
#         5.673038084757683,
#         6.3404543300232925,
#     ),
#     "07": (
#         0.33370812263280486,
#         1.0011243678984145,
#         1.6685406131640244,
#         2.335956858429634,
#         3.003373103695244,
#         3.6707893489608536,
#         4.338205594226463,
#         5.005621839492073,
#         5.673038084757683,
#         6.3404543300232925,
#     ),
#     "08": (
#         0.33370812263280486,
#         1.0011243678984145,
#         1.6685406131640244,
#         2.335956858429634,
#         3.003373103695244,
#         3.6707893489608536,
#         4.338205594226463,
#         5.005621839492073,
#         5.673038084757683,
#         6.3404543300232925,
#     ),
#     "09": (
#         0.5506734696910971,
#         1.6520204090732913,
#         2.7533673484554857,
#         3.8547142878376794,
#         4.956061227219874,
#         6.057408166602068,
#         7.1587551059842625,
#         8.260102045366457,
#         9.36144898474865,
#         10.462795924130845,
#     ),
#     "10": (
#         0.4352714643036585,
#         1.3058143929109756,
#         2.1763573215182923,
#         3.0469002501256095,
#         3.9174431787329262,
#         4.787986107340243,
#         5.65852903594756,
#         6.529071964554877,
#         7.399614893162195,
#         8.27015782176951,
#     ),
# }

# placeholder, we never read these values from the configs
# we should place some restrictions on these to make sure they are never referenced in their current form
pitch_temp, pitchfactor, tilt = -999, -999, -999 

# these were taken from debug martin method
configs = {
    # single axis tracking
    "01": {
        "hub_height": 1.5,
        "pitch": 5,
        "sazm": 180,  # Tracker axis azimuth
        "modulename": "PVmodule",
        "bedsWanted": 3,
        "fixed_tilt_angle": None,
    },
    # single axis tracking
    "02": {
        "hub_height": 2.4,
        "pitch": 5,
        "sazm": 180,
        "modulename": "PVmodule",
        "bedsWanted": 3,
        "fixed_tilt_angle": None,
    }, # single axis tracking
    "03": {
        "hub_height": 2.4,
        "pitch": 5,
        "sazm": 180,
        "modulename": "PVmodule_1mxgap",
        "bedsWanted": 3,
        "fixed_tilt_angle": None,
    },
    "04": {
        "hub_height": 1.5,
        "pitch": 8,
        "sazm": 180,
        "modulename": "PVmodule",
        "bedsWanted": 6,
        "fixed_tilt_angle": None,
    },
    "05": {
        "hub_height": 1.5,
        "pitch": 11,
        "sazm": 180,
        "modulename": "PVmodule",
        "bedsWanted": 9,
        "fixed_tilt_angle": None,
    },
    "06": {
        "hub_height": 1.5,
        "tilt": None,  # fixed,
        "sazm": 180,
        "pitchfactor": 1,
        "modulename": "PVmodule",
        "pitch": pitch_temp * pitchfactor,
        "bedsWanted": 3,
        "fixed_tilt_angle": tilt,
    },
    "07": {
        "hub_height": 2.4,
        "sazm": 180,
        "pitchfactor": 1,
        "pitch": pitch_temp * pitchfactor,
        "modulename": "PVmodule",
        "bedsWanted": 3,
        "fixed_tilt_angle": tilt,
    },
    "08": {
        "hub_height": 2.4,
        "sazm": 180,
        "pitchfactor": 1,
        "pitch": pitch_temp * pitchfactor,
        "modulename": "PVmodule_1mxgap",
        "bedsWanted": 3,
        "fixed_tilt_angle": tilt,
    },
    "09": {
        "hub_height": 1.5,
        "sazm": 180,
        "pitchfactor": 2,
        "pitch": pitch_temp * pitchfactor,
        "modulename": "PVmodule",
        "bedsWanted": 6,
        "fixed_tilt_angle": tilt,
    },
    "10": { # does this want 6 or 7 beds
        "hub_height": 2,
        "sazm": 90,
        "pitch": 8.6,
        "modulename": "PVmodule",
        "bedsWanted": 7,
        "xp": 8,
        "fixed_tilt_angle": 90,
    },
}

def tracking_3_beds(
    dataset: xr.Dataset
) -> xr.Dataset:
    """
    single axis tracking system with 3 beds.

    # SCENARIO 01 02 03 04 #

    Hardcoded from information in "SETUP Description Complete.xlx" values calculated with cw = 2
    """
    left_underpannel_slice = slice(0, 3)
    right_underpannel_slice = slice(7, 10)

    bedA_slice = slice(3,4)
    bedB_slice = slice(4,6)
    bedC_slice = slice(6,7)
    edgetoedge_slice = slice(3,7)

    underpannel_left = dataset.ground_irradiance.isel({"distance":left_underpannel_slice}).mean("distance")
    underpannel_right = dataset.ground_irradiance.isel({"distance":right_underpannel_slice}).mean("distance")
    underpannel = ((underpannel_left + underpannel_right) / 2).rename("underpannel")

    bedA = dataset.ground_irradiance.isel({"distance":bedA_slice}).mean("distance").rename("bedA")
    bedB = dataset.ground_irradiance.isel({"distance":bedB_slice}).mean("distance").rename("bedB")
    bedC = dataset.ground_irradiance.isel({"distance":bedC_slice}).mean("distance").rename("bedC")

    edgetoedge = dataset.ground_irradiance.isel({"distance":edgetoedge_slice}).mean("distance").rename("edgetoedge")

    beds_ds = xr.merge(
        [underpannel, edgetoedge, bedA, bedB, bedC]
    )

    return beds_ds

def tracking_6_beds(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    single axis tracking system with 6 beds

    # SCENARIO 5 #
    """

    left_underpannel_slice = slice(0, 2)
    right_underpannel_slice = slice(8, 10)

    bedA_slice = slice(2, 3)
    bedB_slice = slice(3, 4)
    bedC_slice = slice(4, 5)
    bedD_slice = slice(5, 6)
    bedE_slice = slice(6, 7)
    bedF_slice = slice(7, 8)
    edgetoedge_slice = slice(2, 8)

    underpannel_left = dataset.ground_irradiance.isel({"distance":left_underpannel_slice}).mean("distance")
    underpannel_right = dataset.ground_irradiance.isel({"distance":right_underpannel_slice}).mean("distance")
    underpannel = ((underpannel_left + underpannel_right) / 2).rename("underpannel")

    bedA = dataset.ground_irradiance.isel({"distance":bedA_slice}).mean("distance").rename("bedA")
    bedB = dataset.ground_irradiance.isel({"distance":bedB_slice}).mean("distance").rename("bedB")
    bedC = dataset.ground_irradiance.isel({"distance":bedC_slice}).mean("distance").rename("bedC")
    bedD = dataset.ground_irradiance.isel({"distance":bedD_slice}).mean("distance").rename("bedD")
    bedE = dataset.ground_irradiance.isel({"distance":bedE_slice}).mean("distance").rename("bedE")
    bedF = dataset.ground_irradiance.isel({"distance":bedF_slice}).mean("distance").rename("bedF")

    edgetoedge = dataset.ground_irradiance.isel({"distance":edgetoedge_slice}).mean("distance").rename("edgetoedge")

    beds_ds = xr.merge(
        [underpannel, edgetoedge, bedA, bedB, bedC, bedD, bedE, bedF]
    )

    return beds_ds

def fixed_tilt_vertical_6_beds(
    dataset: xr.Dataset
) -> xr.Dataset:
    """
    vertical fixed tilt system with 6 beds.

    # SCENARIO 10 #

    simply reuses the implementation for the tracking 6 beds because of symmetry.

    for tracking systems, measurement starts at the center of the collector and ends at the center of the collector.
    for fixed systems,    measurement starts at the left side of the left collector and ends on the left side of the next collector.

    for vertical fixed tilt systems the left side of the collector is in the same location (projected on the ground) 
    so we can reuse the math for the tracking system ONLY FOR THIS SCENARIO.

    Image to be included...
    """
    
    return tracking_6_beds(
        dataset=dataset
    )


# TODO
def fixed_tilt_3_beds(
    dataset: xr.Dataset,
) -> xr.Dataset:
    """
    fixed tilt system (latitude tilt) with optimal pitch considerations with 3 beds

    # SCENARIO 6, 7, 8, 9 #

    dataset must contain a pitch datavariable.
    """

    if "pitch" not in dataset.data_vars:
        raise ValueError("dataset not in pitch")

    # magic numbers but hardcoding this is much simpler
    RANGES = [
        ((3.8,     3.8492), (0, 5, 5, 7, 7, 8, 8, 10)),
        ((3.8492,  4.9491), (0, 4, 4, 6, 6, 8, 8, 10)),
        ((4.9491,  6.9355), (0, 3, 3, 5, 5, 8, 8, 10)),
        ((6.9355, 11.5465), (0, 2, 2, 5, 5, 7, 7, 10)),
        ((11.5465, 12.0   ), (0, 1, 1, 4, 4, 7, 7, 10)),
    ]

    def get_indices(pitch: float):
        for (lo, hi), indices in RANGES:
            if lo <= pitch < hi or (hi == 12.0 and lo <= pitch <= hi):
                return indices
        raise ValueError("bad pitch provided, pitch must be in the range [3.8, 12] meters")

    up_left_start_idx, up_left_end_idx, bedA_start_idx, bedA_end_idx, \
    bedB_start_idx, bedB_end_idx, bedC_start_idx, bedC_end_idx = get_indices(pitch)

    left_underpannel_slice = slice(up_left_start_idx, up_left_end_idx)

    bedA_slice = slice(bedA_start_idx, bedA_end_idx)
    bedB_slice = slice(bedB_start_idx, bedB_end_idx)
    bedC_slice = slice(bedC_start_idx, bedC_end_idx)
    edgetoedge_slice = slice(bedA_start_idx, bedC_end_idx)

    underpannel = dataset.ground_irradiance.isel({"distance":left_underpannel_slice}).mean("distance").rename("underpannel")

    bedA = dataset.ground_irradiance.isel({"distance":bedA_slice}).mean("distance").rename("bedA")
    bedB = dataset.ground_irradiance.isel({"distance":bedB_slice}).mean("distance").rename("bedB")
    bedC = dataset.ground_irradiance.isel({"distance":bedC_slice}).mean("distance").rename("bedC")

    edgetoedge = dataset.ground_irradiance.isel({"distance":edgetoedge_slice}).mean("distance").rename("edgetoedge")

    beds_ds = xr.merge(
        [underpannel, edgetoedge, bedA, bedB, bedC]
    )

    return beds_ds


def normalize_per_land_area_fixed_pitch(scenario: int):
    GROUND_AREA_FACTORS = {
        "01":0.261,
        "02":0.261,
        "03":0.522,
        "04":0.417,
        "05":0.579,
        "10":0.453,
    }
    
    if scenario not in GROUND_AREA_FACTORS:
        raise ValueError(f"invalid scenarios provided, only 01, 02, 03, 04, 05, 10, you provided {scenario}")
    
    # normalize

def normalize_per_land_area_variable_pitch(scenario: int, dataset: xr.Dataset):

    TOTAL_MODULE_AREA = 0.104 # acres
    CW = 2 # meters

    # total ground area occupied by array (acres)
    ground_area = TOTAL_MODULE_AREA * dataset.pitch / CW

    # normalize

def normalize_per_kWdc_installed(scenario: int, dataset: xr.Dataset):

    # same for all setups
    INSTALLED_CAPACITY = 80.066 # kWdc

    ### normalize

def postprocessing(scenario: str, zarr_path: Path):
    """
    run postprocessing on model run output zarr stores.

    Parameters
    -----------
    scenario: str
        scenario name 01-10
    zarr_path:
        path to zarr store
    """

    tracking_scenarios = ['01', '02', '03', '04', '05']
    tracking_3_beds_scenarios = ['01', '02', '03', '04']
    fixed_tilt_scenarios = ['06', '07', '08', '09', '10']

    scenario_dataset = xr.open_zarr(zarr_path)

    if scenario in tracking_scenarios:

        if scenario in tracking_3_beds_scenarios:
            beds_dataset = tracking_3_beds(scenario_dataset)
        
        if scenario == "05":
            beds_dataset = tracking_6_beds(scenario_dataset)

    
    if scenario in fixed_tilt_scenarios:

        if scenario == "10":
            # pitch should be in the dataset
            # beds_dataset = fixed_tilt_3_beds(dataset=scenario_dataset, pitch=)
            ...

        if scenario in fixed_tilt_scenarios and scenario != "10":
            ...
    
    else:
        raise ValueError("invalid scenario must be one of 01 02 03 04 05 06 07 08 09 10")




