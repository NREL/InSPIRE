#!/usr/bin/env python
# coding: utf-8

# In[1]:


NREL_API_KEY = None  # <-- please set your NREL API key here
# note you must use "quotes" around your key as it is a string.

if NREL_API_KEY is None:
       NREL_API_KEY = 'DEMO_KEY'  # OK for this demo, but better to get your own key


# In[2]:


import pvlib

metdata, metadata = pvlib.iotools.get_psm3(
    latitude=18.4671, longitude=-66.1185,
    api_key=NREL_API_KEY,
    email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
    names='tmy', map_variables=False)
metadata


# In[3]:


import pvlib

metdata, metadata = pvlib.iotools.get_psm3(
    latitude=18.4671, longitude=-66.1185,
    api_key=NREL_API_KEY,
    email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
    names='tmy', map_variables=True)
metadata


# ## 2. Modeling with bifacial_radiance

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import bifacial_radiance as br


# In[5]:


br.__version__


# In[6]:


import os
from pathlib import Path

testfolder = 'TEMP'

if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)


# In[7]:


radObj = br.RadianceObj('Sim3',path=testfolder)


# In[8]:


# Some of the names changed internally. While bifacial_radiance updates their expected names, we are renaming the values here
metadata['timezone'] = metadata['Time Zone']
metadata['county'] = '-'
metadata['elevation'] = metadata['altitude']
metadata['state'] = metadata['State']
metadata['country'] = metadata['Country']
metdata['Albedo'] = metdata['albedo']


# Use NSRDBWeatherData to enter data the downloaded data in dataframe and dictionary forma for meteorological data and metadata respectively

# In[9]:


#starttime can be 'MM_DD', or 'MM_DD_HH'
metData = radObj.NSRDBWeatherData(metadata, metdata, starttime='11_08_09', endtime='11_08_11',coerce_year=2021)


# In[10]:


metData.datetime  # printing the contents of metData to see how many times got loaded.


# In[11]:


# -- establish tracking angles
hub_height = 1.5
pitch = 5
sazm = 180  # Tracker axis azimuth
modulename = 'PVmodule'
fixed_tilt_angle = None
gcr = 2 / pitch


trackerParams = {'limit_angle':50,
             'angledelta':5,
             'backtrack':True,
             'gcr':gcr,
             'cumulativesky':False,
             'azimuth': sazm,
             'fixed_tilt_angle': fixed_tilt_angle
             }


# In[12]:


trackerdict = radObj.set1axis(**trackerParams)


# In[13]:


radObj.setGround(0.2) 


# In[14]:


radObj.gendaylit1axis()


# In[15]:


module=radObj.makeModule(name=modulename, x=1,y=2)


# In[16]:


sceneDict = {'pitch':pitch, 
             'hub_height': hub_height,
             'nMods': 5,
             'nRows': 2,
             'tilt': fixed_tilt_angle,  
             'sazm': sazm
             }


# In[17]:


trackerdict = radObj.makeScene1axis(module=modulename,sceneDict=sceneDict)


# In[18]:


trackerdict = radObj.makeOct1axis()


# In[19]:


trackerdict = radObj.analysis1axis(customname = 'Module', 
                                   sensorsy=2, modWanted=2,
                                   rowWanted=1)


# In[20]:


trackerdict = radObj.calculateResults(bifacialityfactor=0.7, agriPV=False)


# In[21]:


radObj.CompiledResults


# In[22]:


resolutionGround = 1  #meter. use 1 for faster test runs
numsensors = int((pitch/resolutionGround)+1)
modscanback = {'xstart': 0, 
                'zstart': 0.05,
                'xinc': resolutionGround,
                'zinc': 0,
                'Ny':numsensors,
                'orient':'0 0 -1'}

# Analysis for GROUND
trackerdict = radObj.analysis1axis(customname = 'Ground', 
                                   modscanfront=modscanback, sensorsy=1)


# In[24]:


trackerdict = radObj.calculateResults(bifacialityfactor=0.7, agriPV=True)


# In[25]:


radObj.CompiledResults


# ##  Eploring Accessing the results directly since CompiledResults is failing for agriPV = False
# 
# ## THIS WORKED WITH dev branch up to 4/22, and in the HPC versions we have.
# 

# In[26]:


ResultPVWm2Back = list(radObj.CompiledResults['Grear_mean'])
ResultPVWm2Front = list(radObj.CompiledResults['Gfront_mean'])
ResultGHI = list(radObj.CompiledResults['GHI'])
ResultDHI = list(radObj.CompiledResults['DHI'])
ResultDNI = list(radObj.CompiledResults['DNI'])
ResultPout = list(radObj.CompiledResults['Pout'])
ResultWindSpeed = list(radObj.CompiledResults['Wind Speed'])
ResultPVWm2Back


# In[28]:


# In another wranch?? Thsi hsould have worked
#list(radObj.CompiledResults['Module_temp'])


# In[30]:


keys=list(trackerdict.keys())

groundIrrad = []
temp_air = []
pitch= []
for key in keys:
    groundIrrad.append(trackerdict[key]['Results'][1]['Wm2Front'])
    temp_air.append(trackerdict[key]['temp_air'])
    pitch.append(trackerdict[key]['scene'].sceneDict['pitch'])
    


# In[31]:


results = pd.DataFrame(list(zip(ResultPVWm2Back, ResultPVWm2Front)), columns = ["Back","Front"])


# In[32]:


results['pitch']=trackerdict[key]['scene'].sceneDict['pitch']


# In[33]:


results.to_pickle(results_path)

