import numpy as np
import os
import pandas as pd
import time
import math
from itertools import chain
from itertools import product
import bifacial_radiance
from dask.distributed import Client, LocalCluster, secede
import math
from datetime import datetime, timedelta
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
def simulate_single(daydate=None, wfile=None, system=None, results_folder_fmt=None):    

    loc_name=os.path.split(wfile)[-1].split("_")[1]
    loc_name=loc_name.replace('.csv','')

    # Main configuration
    if system == "mL_utility": #good for comparing with bifacial_vf
        moduletype = 'module_2x1m'
        clearance_height = 1.0
        nMods=21
        iMod = 10
        sensorsx=5
        sim_general_name = 'bifivert_utility'
        tilt = 90
        azimuth = 90 #180 for N-S
        collector_length = 1.0
    elif system == "mP_utility_S": #good for comparing with bifacial_vf
        moduletype = 'module_1x2m'
        clearance_height = 1.0
        nMods=21
        iMod = 10
        sensorsx=5
        sim_general_name = 'bifi-s-tilted_mP'
        azimuth = 180
        collector_length = 2.0
    elif system == "mL_utility_NS": #good for comparing with bifacial_vf
        moduletype = 'module_2x1m'
        clearance_height = 1.0
        nMods=21
        iMod = 10
        sensorsx=5
        sim_general_name = 'bifivert-NS_utility'
        tilt = 90
        azimuth = 180
        collector_length = 1.0
    elif system == "mP_utility_S30": #good for comparing with bifacial_vf
        moduletype = 'module_1x2m'
        clearance_height = 1.0
        nMods=21
        iMod = 10
        sensorsx=5
        sim_general_name = 'bifi-s-tilted-30_mP'
        azimuth = 180
        collector_length = 2.0

     # Verify test_folder exists 
    test_folder = results_folder_fmt.format(loc_name,system,daydate)      
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # Variables to grab from weather file
    #  Strip latitude and longitude from header of csv, lat/long may be called lat/long or degree lat/long
    df = pd.read_csv(wfile)
    if "Latitude" not in df:
        df["Latitude"] = df["Degree latitude"]
    if "Longitude" not in df:
        df["Longitude"] = df["Degree longitude"]
    df = df.head(1)
    lat = float(df["Latitude"].values[0])
    lon = float(df["Longitude"].values[0])

    if system == "mP_utility_S":
        tilt = lat-10
    elif system == "mP_utility_S30":
        tilt = 30

    if spacingType == "single-row":
        nRows = 1
        iRow = 1
        pitch = 100
    elif spacingType == "single-spacing":
        nRows = 7
        iRow = 4
        pitch = 7
    elif spacingType == "latitude-dependent":
        nRows = 7
        iRow = 4
        if system == "mL_utility":
            m_v=-6.87e-4
            b_v=0.155
            pitch = 1/((m_v*lat)+b_v)
        elif system == "mP_utility_S":
            P=-0.560
            k=0.133
            a_0=40.2
            GCR_0=0.70
            pitch = 2/((P/(1+np.exp(-k*(lat-a_0))))+GCR_0)
        elif system == "mL_utility_NS":
            m_v=-6.87e-4
            b_v=0.155
            pitch = 1/((m_v*lat)+b_v)
        elif system == "mP_utility_S30":
            P=-0.560
            k=0.133
            a_0=40.2
            GCR_0=0.70
            pitch = 2/((P/(1+np.exp(-k*(lat-a_0))))+GCR_0)
    gcr = collector_length/pitch
    print(spacingType+" pitch = "+str(pitch)+", gcr = "+str(gcr))

    #Main Variables needed throughout
    sensorsy=8
    hpc = True
    cumulativesky = False
    limit_angle = 90
    backtrack = False

    #Grab daydate and format for what is needed for readWeatherFile
    date=daydate.replace("_","-")
    starttime="20"+date+"_0300"
    endtime="20"+date+"_2200"

    #START SIMULATION
    sim_name = '_'+loc_name+'_lat='+str(lat)+'_long='+str(lon)+'_'+sim_general_name+'_'+moduletype+'_tilt='+str(round(tilt,1))+'_imod='+str(round(iMod))+'_pitch='+str(round(pitch,1))
    demo = bifacial_radiance.RadianceObj(sim_name,str(test_folder),hpc=hpc)
    if FullYear == True:
        metdata = demo.readWeatherFile(wfile,starttime=starttime,endtime=endtime,source='sam',label='center',coerce_year=2023)
    else:
        metdata = demo.readWeatherFile(wfile,starttime=starttime,endtime=endtime,source='sam',label='center',coerce_year=2023)
    demo.setGround()
    sceneDict = {'tilt':tilt,'pitch':pitch,'clearance_height':clearance_height,'azimuth':azimuth, 'nMods': nMods, 'nRows': nRows} 
    trackerdict = demo.set1axis(azimuth=azimuth, fixed_tilt_angle=tilt, gcr=gcr, backtrack = backtrack, cumulativesky = cumulativesky)
    trackerdict = demo.gendaylit1axis()
    trackerdict = demo.makeScene1axis(module=moduletype,sceneDict=sceneDict) 
    trackerdict = demo.makeOct1axis(customname = sim_name) 
    demo.analysis1axis(customname = sim_name, sensorsx=sensorsx, sensorsy=sensorsy, modWanted=iMod, rowWanted=iRow) 

    #scene.saveImage() #try this with gendaylit for one hour to see array
    # Make a color render and falsecolor image of the scene.
    #analysis.makeImage('side.vp')
    #analysis.makeFalseColor('side.vp')
    results = 1

    return results


def run_simulations_dask(daylist,wfiles,systems,kwargs,hpc):
    # Create client

    client = start_dask(hpc)
    
    # Iterate over inputs
    futures = []
    
    # Add Iterations HERE

    for wfile in wfiles:
        for system in systems:
            for daydate in daylist:
                futures.append(client.submit(simulate_single,daydate=daydate,wfile=wfile,system=system,**kwargs)) #Creates jobID for future workers to be called on

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

    print(">>>>>>>>>>>>>>>>>> STARTING HERE !")
    print(datetime.now())
    sim_start_time=datetime.now()
    
    FullYear = True
    spacingType = "latitude-dependent" #single-row; latitude-dependent; single-spacing
    
    if FullYear:
        start = datetime.strptime("01-01-2023", "%d-%m-%Y")
        end = datetime.strptime("31-12-2023", "%d-%m-%Y")
    else:
        start = datetime.strptime("01-11-2023", "%d-%m-%Y")
        end = datetime.strptime("30-11-2023", "%d-%m-%Y")
    date_generated = [start + timedelta(days=x) for x in range(0, (end-start).days)]
    daylist = []
    for date in date_generated:
        daylist.append(date.strftime("%y_%m_%d"))
    # loop doesn't add last day :
    daylist.append('23_12_31')
    print(daylist) #check no repeated elements
    

    local = {'manager': 'local',
        'n_workers': 32,
        'threads_per_worker': 1, # Number of CPUs
        }
    kestrel = {
        'manager': 'slurm',
        'n_jobs': 4,  # Number of nodes used for parallel processing
        'cores': 104, #This is the total number of threads in all workers
        'memory': '256GB',
        'account': 'pvsoiling',
        'queue': 'standard',
        'walltime': '6:59:00', 
        'processes': 102, #This is the number of workers
        #'interface': 'lo'
        #'job_extra_directives': ['-o ./logs/slurm-%j.out'],
        }
    
    wfiles=[]
    directory="/home/etonita/WeatherFiles/All-TMY-and-CWEC/"
    for path, subdirs, files in os.walk(directory):
        for filename in files:
            f = os.path.join(path, filename)
            wfiles.append(f)

    print(wfiles)

    systems=['mL_utility_NS','mP_utility_S30']

    now=datetime.now()
    results_path = "/scratch/etonita/Bifacial_Radiance_Simulations/HourlyNSS30"+"_"+now.strftime('%Y-%m-%d_%Hh%M')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    
    results_folder_fmt = results_path+"/{}/{}/DAY_{}"

    # Define inputs    
    kwargs = {
        'results_folder_fmt': results_folder_fmt
    }


    # Pass variables being looped on, and kwargs
    run_simulations_dask(daylist, wfiles, systems, kwargs, hpc=kestrel)

    print("*********** DONE ************")
    print(datetime.now())
    sim_end_time=datetime.now()
    print("Total time to run simulation = "+ str(sim_end_time - sim_start_time))
    