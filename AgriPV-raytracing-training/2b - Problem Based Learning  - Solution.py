#!/usr/bin/env python
# coding: utf-8

# # 2b. SOLUTION Problem Based Learning: Pyranometers at BARN
# 
# 
# <div class="alert alert-block alert-warning"> <b>NOTE</b> This is just one way we can solve the PBL. Try it yourself before reading this solution. </div>
# 
# 
# PREMISE: With what you know, model the irradiance accross the 5 Pyranometer sensors in the bifacial field located beewteen row 7 and 8.
# 

# Couple measurements:
# * hub_height = 1.5 m
# * pitch = 5.7 m
# * modules are in 1-up portrait
# * Field size is 10 rows x 20 modules
# * Module are 2x1 m
# * Pyranometers start 16in off east-way from the center of row 7, module 14
#     
# Suggestions:
# * Do an hourly simulation. To calculate the tracker angle, look at the function and documentation of [`getSingleTimestampTrackerAngle`](https://bifacial-radiance.readthedocs.io/en/latest/generated/bifacial_radiance.RadianceObj.getSingleTimestampTrackerAngle.html#bifacial_radiance.RadianceObj.getSingleTimestampTrackerAngle)
# 

# In[1]:


import os
from pathlib import Path

testfolder = str(Path().resolve().parent / 'TEMP' /  'Barn')

if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)


# In[2]:


import bifacial_radiance as br
import numpy as np
import pandas as pd


# In[3]:


# This information helps with debugging and getting support :)
import sys, platform
print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("bifacial_radiance version ", br.__version__)


# In[6]:


simulationname = 'tutorial_1'

# Location:
lat = 39.7555
lon = -105.2211

# Scene Parameters:
azimuth_ang=90 # Facing east

# MakeModule Parameters
moduletype='PVmod'
module_x = 1 # m
module_y = 2 # m. slope we will measure
# if x > y, landscape. if x < y, portrait

# SceneDict Parameters
pitch = 5.7 # m
hub_height = 1.5 # m  
nMods = 20 # six modules per row.
nRows = 10  # 3 row


# In[7]:


barn = br.RadianceObj(simulationname,path = testfolder)  
epwfile = barn.getEPW(lat, lon) 


# In[9]:


module=barn.makeModule(name=moduletype,x=module_x,y=module_y)


# In[10]:


#Determine Hour to model
#Valid options: mm_dd, mm_dd_HH, mm_dd_HHMM, YYYY-mm-dd_HHMM
metdata = barn.readWeatherFile(epwfile, coerce_year=2021, starttime='2021-08-30_09', endtime='2021-08-30_09')


# In[11]:


metdata.__dict__


# In[12]:


metdata.dni


# In[13]:


metdata.albedo


# In[15]:


barn.setGround(metdata.albedo) 


# In[16]:


module


# In[18]:


metdata.datetime


# In[17]:


#Calculate GCR
#gcr = CW / rtr
gcr = module.sceney / pitch 
gcr


# In[19]:


timeindex = metdata.datetime.index(pd.to_datetime('2021-08-30 09:00:00-0700'))  # Make this timezone aware, use -5 for EST.
timeindex


# In[20]:


timeindex = metdata.datetime.index(pd.to_datetime('2021-08-30 9:0:0 -7'))  # Make this timezone aware, use -5 for EST.
timeindex


# In[24]:


tilt = barn.getSingleTimestampTrackerAngle(metdata=metdata, timeindex=0, gcr=0.33, axis_tilt=0, limit_angle=50, backtrack=True)
tilt


# In[25]:


barn.gendaylit(timeindex=0)  


# In[26]:


sceneDict = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods, 'nRows': nRows}  
scene = barn.makeScene(module=moduletype, sceneDict=sceneDict) 


# In[27]:


octfile = barn.makeOct()


# In[31]:


get_ipython().system('rvu -vf views\\front.vp -e .01 -pe 0.4 -vp 0 -20 50 -vd 0 0.7 -0.7 tutorial_1.oct')


# In[ ]:



