import numpy as np
import os
import pandas as pd
import time
import math
from itertools import chain
from itertools import product
import bifacial_radiance as br
from dask.distributed import Client, LocalCluster, secede
import math
import datetime
from timeit import default_timer as timer
from time import sleep
from rex import NSRDBX
import pytz
import pickle
import bifacialvf
import shutil

def start_dask(hpc=None):
    """
    Starts a dask cluster for parallel processing.

    Parameters
    ----------
    hpc : dict
        Dictionary containing dask hpc settings (see examples below).

    Examples
    --------
    Local cluster:

    .. code-block:: python

        hpc = {'manager': 'local',
               'n_workers': 1,
               'threads_per_worker': 8,
               'memory_limit': '10GB'}

    SLURM cluster:

    .. code-block:: python

        kestrel = {
            'manager': 'slurm',
            'n_jobs': 1,  # Max number of nodes used for parallel processing
            'cores': 104,
            'memory': '256GB',
            'account': 'pvsoiling',
            'walltime': '4:00:00',
            'processes': 52,
            'local_directory': '/tmp/scratch',
            'job_extra_directives': ['-o ./logs/slurm-%j.out'],
            'death_timeout': 600,}

    Returns
    -------
    client : dask.distributed.Client
        Dask client object.
    """
    if hpc is None:
        cluster = LocalCluster()
    else:
        manager = hpc.pop('manager')

        if manager == 'local':
            cluster = LocalCluster(**hpc)
        elif manager == 'slurm':
            from dask_jobqueue import SLURMCluster
            n_jobs = hpc.pop('n_jobs')
            cluster = SLURMCluster(**hpc)
            cluster.scale(jobs=n_jobs)

    client = Client(cluster)
    print('Dashboard:', client.dashboard_link)
    client.wait_for_workers(n_workers=1)

    return client

# Run simulation using the given timestamp and wavelength
def simulate_single(df_tmy = None, meta_dict = None, gid = None, setup = None,
             startdate=None, rootPath=None):

    startdatestr = str(startdate).replace(':','_').replace(' ','__')

    #startdate = None
    #enddate = None
    simpath = f'{state}_{gid}_setup_{setup}_{startdatestr}'
    if rootPath is None:
        path = os.path.join(state,str(setup),simpath)
    else:
        path = os.path.join(rootPath,state,str(setup),simpath)
    results_path = os.path.join(path, 'results.pkl')

    #os.path.isfile(path)
    #Check if simulation has already completed
    print("Current path", path)

    if os.path.isfile(results_path):
        print("***** SIM Done for ", simpath)
        print("Results are: ", results)
        
        return 1

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    alb = 0.2
    radObj = br.RadianceObj(simpath,path)
    radObj.setGround(alb) 

    enddate = startdate + datetime.timedelta(hours=23)

    metData = radObj.NSRDBWeatherData(meta_dict, df_tmy, starttime=startdate, 
                                      endtime=enddate, coerce_year=2023)
    
    # Tracker Projection of half the module into the ground, 
    # for 1-up module in portrait orientation
    # Assuming still 1 m for fixed-tilt systems even if a bit less 
    # VERTICAL setup changes xp to just a margin 20 cm for protection of the modules.
    xp = 10 
    y = 2
    solposAzi = metData.solpos['azimuth'][0]
    solposZen = metData.solpos['zenith'][0]
    timezonesave = metData.timezone

    tilt = np.round(metData.latitude)
    if tilt > 40.0:
        tilt = 40.0
    
    DD = bifacialvf.vf.rowSpacing(beta = tilt, 
                                  sazm=180, lat = metData.latitude, 
                                  lng = metData.longitude, 
                                  tz = metData.timezone, 
                                  hour = 9, 
                                  minute = 0.0)
    if (DD <= 0) or (DD > 3.725):
        DD = 3.725
        print("Cannot find ideal pitch for location, setting D to 3.725")

    normalized_pitch = DD + np.cos(np.round(metData.latitude) / 180.0 * np.pi)
    pitch_temp = normalized_pitch*y
    print("PITCH_temp ", pitch_temp)

    if setup == 1:
        hub_height = 1.5
        pitch = 5
        sazm = 180  # Tracker axis azimuth
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = None
    if setup == 2:
        hub_height = 2.4
        pitch = 5
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = None
    if setup == 3:
        hub_height = 2.4
        pitch = 5
        sazm = 180
        modulename = 'PVmodule_1mxgap'
        bedsWanted = 3
        fixed_tilt_angle = None
    if setup == 4:
        hub_height = 1.5
        pitch = 8
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 6
        fixed_tilt_angle = None
    if setup == 5:
        hub_height = 1.5
        pitch = 11
        sazm = 180
        modulename = 'PVmodule'
        bedsWanted = 9
        fixed_tilt_angle = None
    if setup == 6:
        hub_height = 1.5
        #tilt = None # fixed
        sazm = 180
        pitchfactor = 1
        modulename = 'PVmodule'
        pitch = pitch_temp * pitchfactor
        bedsWanted = 3
        fixed_tilt_angle = tilt
    if setup == 7:
        hub_height = 2.4 
        sazm = 180
        pitchfactor = 1
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule'
        bedsWanted = 3
        fixed_tilt_angle = tilt
    if setup == 8:
        hub_height = 2.4 
        sazm = 180
        pitchfactor = 1
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule_1mxgap'
        bedsWanted = 3
        fixed_tilt_angle = tilt
    if setup == 9:
        hub_height = 1.5 
        sazm = 180
        pitchfactor = 2
        pitch = pitch_temp * pitchfactor
        modulename = 'PVmodule'
        bedsWanted = 6
        fixed_tilt_angle = tilt
    if setup == 10:
        hub_height = 2 
        sazm = 90
        pitch = 8.6 
        modulename = 'PVmodule'
        bedsWanted = 7
        xp = 8
        fixed_tilt_angle = 90

    # TILT & PITCH CALCULATION HERE

    gcr = 2/pitch

    # -- establish tracking angles
    trackerParams = {'limit_angle':50,
                     'angledelta':5,
                     'backtrack':True,
                     'gcr':gcr,
                     'cumulativesky':False,
                     'azimuth': sazm,
                     'fixed_tilt_angle': fixed_tilt_angle,
                     }

    trackerdict = radObj.set1axis(**trackerParams)
    
    # -- generate sky   
    trackerdict = radObj.gendaylit1axis()
    print(trackerdict)
    print("LEN TRACKERDICT", len(trackerdict.keys()))
    try:
        tracazm = trackerdict[list(trackerdict.keys())[0]]['surf_azm']
        tractilt = trackerdict[list(trackerdict.keys())[0]]['surf_tilt']
    except:
        print("Issue with tracazm/tractilt on trackerdict for ", path )
        tracazm = np.NaN
        tractilt = np.NaN  

    sceneDict = {'pitch':pitch, 
                 'hub_height': hub_height,
                 'nMods': 19,
                 'nRows': 7,
                'tilt': fixed_tilt_angle,  
                'sazm': sazm
                 }

    modWanted = 10
    rowWanted = 4

    trackerdict = radObj.makeScene1axis(module=modulename,sceneDict=sceneDict)

    # -- build oct file
    trackerdict = radObj.makeOct1axis()

    # -- run analysis
    # Analysis for Module
    trackerdict = radObj.analysis1axis(trackerdict, customname = 'Module',
                                       sensorsy=9, modWanted=modWanted,
                                       rowWanted=rowWanted)

    try:
        trackerdict = radObj.calculateResults(bifacialityfactor=0.7, agriPV=False)
    except:
        print("**Error on trackerdict WHY!, skipping", gid, startdate)
        print("Trackerdict error path: " , results_path)
        print("TRACKERDICT Print:", radObj.trackerdict)      
        return 0

    ResultDatetime = list(radObj.CompiledResults['timestamp']) 
    ResultPVWm2Back = list(radObj.CompiledResults['Grear_mean'])
    ResultPVWm2Front = list(radObj.CompiledResults['Gfront_mean'])
    ResultGHI = list(radObj.CompiledResults['GHI'])
    ResultDNI = list(radObj.CompiledResults['DNI'])
    ResultDHI = list(radObj.CompiledResults['DHI'])
    ResultPout = list(radObj.CompiledResults['Pout'])
    # Commenting out due to using version 0.4.2+236.g6801d3d
    # ResultModuleTemp = list(radObj.CompiledResults['Module_temp'])
    ResultWindSpeed = list(radObj.CompiledResults['Wind Speed'])

    # Modify modscanfront for Ground
    resolutionGround = 0.1  # use 1 for faster test runs
    numsensors = int((pitch/resolutionGround)+1)
    modscanfront = {'xstart': 0, 
                    'zstart': 0.05,
                    'xinc': resolutionGround,
                    'zinc': 0,
                    'Ny':numsensors,
                    'orient':'0 0 -1'}

    # Analysis for GROUND
    trackerdict = radObj.analysis1axis(trackerdict, customname = 'Ground',
                                       modWanted=modWanted, rowWanted=rowWanted,
                                        modscanfront=modscanfront, sensorsy=1)
 
    keys=list(trackerdict.keys())

    ResultGroundIrrad = []
    ResultTemp = []
    for key in keys:
        ResultGroundIrrad.append(trackerdict[key]['Results'][1]['Wm2Front'])
        ResultTemp.append(trackerdict[key]['temp_air'])

    # Cleanup of Front files from the Ground simulation
    filesall = os.listdir('results')
    filestoclean = [e for e in filesall if e.endswith('_Back.csv')]
    for cc in range(0, len(filestoclean)):
        filetoclean = filestoclean[cc]
        os.remove(os.path.join('results', filetoclean))

    results = pd.DataFrame(list(zip(ResultDatetime, ResultGHI, ResultDNI, ResultDHI, 
                                    ResultTemp, ResultWindSpeed, 
                                    ResultPVWm2Back, ResultPVWm2Front,
                                    ResultPout, ResultGroundIrrad #ResultModuleTemp, 
                                    )), 
                                    columns=["Timestamp", "GHI", "DNI", "DHI", "AirTemp", 
                                             "WindSpeed", "PVWm2Back", "PVWM2Front",
                                                  "PVPowerOutputW", "GroundIrrad" # "PVModuleTemp",
                                                  ])
    results["gid"] = gid
    results["setup"] = setup
    results["latitude"] = metData.latitude
    results["longitude"] = metData.longitude
    results["pitch"] = trackerdict[key]['scene'].sceneDict['pitch']

    # save to folder    
    results.to_pickle(results_path)
    print("Results pickled!")

    if os.path.isfile(results_path):
        # Verifies CSV file was created, then deletes unneeded files.
        for clean_up in os.listdir(path):
            if not clean_up.endswith('results.pkl'):
                clean_upfile = os.path.join(path, clean_up)
                if os.path.isfile(clean_upfile):    
                    os.remove(clean_upfile)
                else:
                    shutil.rmtree(clean_upfile)
    print("Results len ", len(results), " type ", type(results))
    print("All other files cleaned!")

    print("***** SIM Done for ", simpath, len(results), " \n Results: ", results)

    return 1