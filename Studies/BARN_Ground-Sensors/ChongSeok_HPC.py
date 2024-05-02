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
import pytz
import pickle
import bifacialvf
import shutil
# import subprocess

# rc = subprocess.call("/home/etonita/BasicSimulations/start_script.sh")

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
            'account': 'inspire',
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
def simulate_single(weatherfile, startdate, results_path, smaller_sim):

    startdatestr = str(startdate).replace(':','_').replace(' ','__')
    simpath = f'{startdatestr}'
    path = os.path.join(results_path,simpath)
    #Check if simulation has already completed
    print("Current path", path)

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    radObj = br.RadianceObj(simpath,path)
    if smaller_sim:
        enddate = startdate + datetime.timedelta(hours=2)
    else:
        enddate = startdate + datetime.timedelta(hours=23)
    
    metdata = radObj.readWeatherFile(weatherfile, coerce_year=2023, source='SAM', 
                                     starttime=startdate, endtime=enddate, 
                                     label='center')

    modulename = 'PVmodule'

    sceneDict = {'pitch':5.7, 
                 'hub_height': 1.5,
                 'nMods': 20,
                 'nRows': 10,
                'sazm': 180
                 }
    
    # TILT & PITCH CALCULATION HERE
    gcr = 2/sceneDict['pitch']

    # -- establish tracking angles
    trackerParams = {'limit_angle':50,
                     'backtrack':True,
                     'gcr':gcr,
                     'cumulativesky':False, # doing hourly sims.
                     'azimuth': sazm, # N-S : 180
                     }

    trackerdict = radObj.set1axis(**trackerParams)
    radObj.setGround() 
    trackerdict = radObj.gendaylit1axis()
    trackerdict = radObj.makeScene1axis(module=modulename,sceneDict=sceneDict)
    trackerdict = radObj.makeOct1axis()

    # Analysis for the Ground
    sensorsposition_x = [0.4254, 1.6254, 2.8254, 4.0254, 5.2254] # original sensor positons on meters
    inchtom = 0.0254

    if smaller_sim:
        simsnum = 2
    else:
        simsnum = len(sensorsposition_x)
    
    for sensor in range(0, simsnum):

        offset = offsets[ii]*inchtom  # passing to meters.
        xstart = sensorsposition_x[sensor]
    
        modscanground = {'xstart': xstart-2*inchtom, # an inch east of 0.4 meter starting point
                        'zstart': 0.05,
                        'xinc': 0,                                                                                                                                                                                                                                                        
                        'zinc': 0,
                        'yinc': 1*inchtom
                        'Ny':5,
                        'orient':'0 0 -1'}
        # counts from bottom right ...
        trackerdict = radObj.analysis1axis(trackerdict, customname = 'Ground_S'+str(sensor+1), modscanfront=modscanground)

    print("***** SIM Done for ", simpath)

    return 1

def run_simulations_dask(weatherfile, startdates, 
                         results_path, hpc, smaller_sim):
    # THIS DOES:
    # Create the dask client, submits the jobs to each core, and saves results.
    
    # Create client      
    client = start_dask(hpc)
    print("dask client started")

    # Iterate over inputs
    futures = []  

    # Add Iterations HERE
    for dd in range(0, len(startdates)):
        #print("setup: ", setup, ", startdate: ", dd)
        startdate = startdates[dd]
        futures.append(client.submit(simulate_single, weatherfile=weatherfile,
                                    startdate=startdate, results_path=results_path,
                                    smaller_sim=smaller_sim)) 

    # Get results for all simulations
    res = client.gather(futures)
    
    # Close all dask workers and scheduler
    try:
    	client.shutdown()
    except:
        pass

    # Close client
    client.close()

    res = 'FINISHED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    return res

if __name__ == "__main__":

    # MAIN DOES:
    # Load weather data
    # Define the variable we will be looping through, in this case startdates
    # Call run_simulations_dask
    # Time the run

    print(">>>>>>>>>>>>>>>>>> STARTING HERE !")
    print(datetime.datetime.now())
    sim_start_time=datetime.datetime.now()
    
    smaller_sim = True
    
    if smaller_sim:
        start = datetime.datetime(2023, 9, 26, 0, 0)
        end = datetime.datetime(2023, 9, 26, 0, 0)
    else:
        start = datetime.datetime(2023, 8, 28, 0, 0)
        end = datetime.datetime(2023, 11, 3, 0, 0)
    
    daylist = []
    while start <= end:
        daylist.append(start)
        start += datetime.timedelta(days=1)

    local = {'manager': 'local',
        'n_workers': 32,
        'threads_per_worker': 1, # Number of CPUs
        }
    kestrel = {
        'manager': 'slurm',
        'n_jobs': 4,  # Number of nodes used for parallel processing #1
        'cores': 104, #This is the total number of threads in all workers was #26
        'memory': '256GB',
        'account': 'inspire',
        'queue': 'standard', #'debug'
        'walltime': '8:00:00', 
        'processes': 102, #This is the number of workers #24
        #'interface': 'lo'
        #'job_extra_directives': ['-o ./logs/slurm-%j.out'],
        }

    now=datetime.datetime.now()
    
    results_path = "/scratch/sayala/chongseok"+"_"+now.strftime('%Y-%m-%d_%Hh%M')
    
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    print("Bifacial_radiance version ", br.__version__)
    print("Pandas version ", pd.__version__)
    print("Numpy version ", np.__version__)

    starttimer = timer()

    weatherfile = r'/home/sayala/WeatherFiles/PSM3_15T.csv'

    run_simulations_dask(weatherfile = weatherfile, 
                        startdates = daylist, 
                        results_path = results_path, 
                        hpc=kestrel, smaller_sim = smaller_sim)


    print("*********** DONE ************")

    # =========== Simulation RunTime Calculation ===========
    stop = timer()
    runTime = round(stop-starttimer,2)
    min = int(runTime//60)
    sec = int(round((runTime/60 - min)*60,0))
    print('=======================================')
    print(f'Simulation Run Time: {min:02}:{sec:02}')
    print('=======================================')