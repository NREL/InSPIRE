import pandas as pd
import numpy as np
import xarray as xr


DISTANCES = {
    "01": (
        0.2502810919746036,
        0.7508432759238107,
        1.251405459873018,
        1.751967643822225,
        2.2525298277714323,
        2.753092011720639,
        3.2536541956698466,
        3.7542163796190535,
        4.2547785635682605,
        4.755340747517468,
    ),
    "02": (
        0.2502810919746036,
        0.7508432759238107,
        1.251405459873018,
        1.751967643822225,
        2.2525298277714323,
        2.753092011720639,
        3.2536541956698466,
        3.7542163796190535,
        4.2547785635682605,
        4.755340747517468,
    ),
    "03": (
        0.2502810919746036,
        0.7508432759238107,
        1.251405459873018,
        1.751967643822225,
        2.2525298277714323,
        2.753092011720639,
        3.2536541956698466,
        3.7542163796190535,
        4.2547785635682605,
        4.755340747517468,
    ),
    "04": (
        0.4004497471593658,
        1.2013492414780975,
        2.002248735796829,
        2.8031482301155606,
        3.6040477244342926,
        4.404947218753024,
        5.205846713071756,
        6.006746207390488,
        6.807645701709219,
        7.608545196027951,
    ),
    "05": (
        0.5561802043880081,
        1.6685406131640241,
        2.7809010219400405,
        3.8932614307160565,
        5.005621839492073,
        6.117982248268089,
        7.230342657044105,
        8.342703065820121,
        9.455063474596137,
        10.567423883372154,
    ),
    "06": (
        0.33370812263280486,
        1.0011243678984145,
        1.6685406131640244,
        2.335956858429634,
        3.003373103695244,
        3.6707893489608536,
        4.338205594226463,
        5.005621839492073,
        5.673038084757683,
        6.3404543300232925,
    ),
    "07": (
        0.33370812263280486,
        1.0011243678984145,
        1.6685406131640244,
        2.335956858429634,
        3.003373103695244,
        3.6707893489608536,
        4.338205594226463,
        5.005621839492073,
        5.673038084757683,
        6.3404543300232925,
    ),
    "08": (
        0.33370812263280486,
        1.0011243678984145,
        1.6685406131640244,
        2.335956858429634,
        3.003373103695244,
        3.6707893489608536,
        4.338205594226463,
        5.005621839492073,
        5.673038084757683,
        6.3404543300232925,
    ),
    "09": (
        0.5506734696910971,
        1.6520204090732913,
        2.7533673484554857,
        3.8547142878376794,
        4.956061227219874,
        6.057408166602068,
        7.1587551059842625,
        8.260102045366457,
        9.36144898474865,
        10.462795924130845,
    ),
    "10": (
        0.4352714643036585,
        1.3058143929109756,
        2.1763573215182923,
        3.0469002501256095,
        3.9174431787329262,
        4.787986107340243,
        5.65852903594756,
        6.529071964554877,
        7.399614893162195,
        8.27015782176951,
    ),
}

# in sam config, subarray1_track mode
# 0: Fixed tilt (no tracking).
# 1: Single-axis tracking.
# 2: Two-axis tracking.

# ONLY CARE ABOUT FIRST 3 FOR NOW
# ASK SILVANA ABOUT HOW TO DEAL WITH THESE AND PROGRAMATICALLY EXTRACT FROM SAM CONFIG FILE, ALSO WANT TO SET 
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
    # "04": {
    #     "hub_height": 1.5,
    #     "pitch": 8,
    #     "sazm": 180,
    #     "modulename": "PVmodule",
    #     "bedsWanted": 6,
    #     "fixed_tilt_angle": None,
    # },
    # "05": {
    #     "hub_height": 1.5,
    #     "pitch": 11,
    #     "sazm": 180,
    #     "modulename": "PVmodule",
    #     "bedsWanted": 9,
    #     "fixed_tilt_angle": None,
    # },
    # "06": {
    #     "hub_height": 1.5,
    #     "tilt": None,  # fixed,
    #     "sazm": 180,
    #     "pitchfactor": 1,
    #     "modulename": "PVmodule",
    #     "pitch": pitch_temp * pitchfactor,
    #     "bedsWanted": 3,
    #     "fixed_tilt_angle": tilt,
    # },
    # "07": {
    #     "hub_height": 2.4,
    #     "sazm": 180,
    #     "pitchfactor": 1,
    #     "pitch": pitch_temp * pitchfactor,
    #     "modulename": "PVmodule",
    #     "bedsWanted": 3,
    #     "fixed_tilt_angle": tilt,
    # },
    # "08": {
    #     "hub_height": 2.4,
    #     "sazm": 180,
    #     "pitchfactor": 1,
    #     "pitch": pitch_temp * pitchfactor,
    #     "modulename": "PVmodule_1mxgap",
    #     "bedsWanted": 3,
    #     "fixed_tilt_angle": tilt,
    # },
    # "09": {
    #     "hub_height": 1.5,
    #     "sazm": 180,
    #     "pitchfactor": 2,
    #     "pitch": pitch_temp * pitchfactor,
    #     "modulename": "PVmodule",
    #     "bedsWanted": 6,
    #     "fixed_tilt_angle": tilt,
    # },
    # "10": {
    #     "hub_height": 2,
    #     "sazm": 90,
    #     "pitch": 8.6,
    #     "modulename": "PVmodule",
    #     "bedsWanted": 7,
    #     "xp": 8,
    #     "fixed_tilt_angle": 90,
    # },
}

def tracking_tilt_3_beds(
    irradiance_df: pd.DataFrame,

    underpanel_left_end: float,
    cw: float,
    pitch: float
) -> pd.DataFrame:
    # Tracking TILT, 3 beds
    xp = cw/2 # Projection of panel on P.
    u = int(np.ceil(10*underpanel_left_end/pitch)) # underpanel limit integer box
    b = 10-u
    # Three testbeds:
    e2e = b-u # dimensions of edge to edge
    bA = int(np.ceil(e2e/3.0))
    bC = int(bA)
    bB = int(e2e-bA-bC)

    underpanel = irradiance_df.iloc[:,0:u].join(irradiance_df.iloc[:,b:10]).mean(axis=1)
    edgetoedge = irradiance_df.iloc[:,u:b].mean(axis=1)
    bedA = irradiance_df.iloc[:,u:u+bA].mean(axis=1)
    bedB = irradiance_df.iloc[:,u+bA:u+bA+bB].mean(axis=1)
    bedC = irradiance_df.iloc[:,u+bA+bB:b].mean(axis=1)

    return pd.DataFrame({
        "underpanel":underpanel,
        "edgetoedge":edgetoedge,
        "bedA":bedA,
        "bedB":bedB,
        "bedC":bedC,
    })


def testbeds_irradiance(conf: str, irradiance_df: pd.DataFrame) -> pd.DataFrame:

    settings = configs[conf]

    # this isnt fized, what should we be using?
    cw = 2
    pitch = settings["pitch"]

    underpanel_left_start = 0
    underpanel_left_end = cw / 2  # u
    underpanel_right_start = pitch - cw / 2  # b
    underpanel_right_end = pitch

    # sam ground irradiance outputs have 10 distances
    dx = pitch/10


    if settings["fixed_tilt_angle"] is None and settings["bedsWanted"] == 3:
        postprocess_df = tracking_tilt_3_beds(
            irradiance_df=irradiance_df,
            cw=cw,
            underpanel_left_end=underpanel_left_end,
            pitch=pitch
        )
    
    else:
        raise ValueError("configuration not implemented yet")

    return postprocess_df


def output_template(pysam_preprocess_ds: xr.Dataset) -> xr.Dataset:
    # Define the coordinates
    coords = {
        'latitude': pysam_preprocess_ds.latitude.values,
        'longitude': pysam_preprocess_ds.longitude.values,
        'distance_index': np.arange(10),  # Example 10 distance indices
        'time': pysam_preprocess_ds.time.values,
    }
    
    # Define the shape of each dimension
    dims = {
        'latitude': len(coords['latitude']),
        'longitude': len(coords['longitude']),
        'distance_index': len(coords['distance_index']),
        'time': len(coords['time']),
    }
    
    # Define the data variables with appropriate shapes
    data_vars = {
        'temp_air': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'dew_point': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'dhi': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'dni': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'ghi': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'albedo': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'pressure': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'wind_direction': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'wind_speed': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'relative_humidity': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        
        'ground_irradiance': (
            ['time', 'distance_index', 'latitude', 'longitude'],
            np.empty((dims['time'], dims['distance_index'], dims['latitude'], dims['longitude']))
        ),
        
        'underpanel': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'edgetoedge': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'bedA': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'bedB': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
        'bedC': (['latitude', 'longitude', 'time'], np.empty((dims['latitude'], dims['longitude'], dims['time']))),
    }
    
    # Return the dataset
    return xr.Dataset(data_vars=data_vars, coords=coords)
