#!/usr/bin/env python
# coding: utf-8

# # 2 - Model on bifacial_radiance
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

testfolder = os.path.join('TEMP', 'bifacial_radiance')

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


# In[4]:


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


# In[5]:


barn = br.RadianceObj(simulationname,path = testfolder)  


# In[6]:


weatherfile = str(Path().resolve().parent.parent / 'WeatherFiles' /  'PSM3_15T.csv')


# In[11]:


#starttime='2021-08-30_09', endtime='2021-08-30_09'
#Valid options: mm_dd, mm_dd_HH, mm_dd_HHMM, YYYY-mm-dd_HHMM
#metdata = barn.readWeatherFile(weatherfile, coerce_year=2023, source='SAM', starttime='2023-08-29_09', endtime='2023-08-29_09', label='right')
metdata = barn.readWeatherFile(weatherfile, coerce_year=2023, source='SAM', starttime='2023-08-29', endtime='2023-08-29', label='center')


# In[12]:


module=barn.makeModule(name=moduletype,x=module_x,y=module_y)


# In[13]:


barn.metdata.__dict__


# In[14]:


barn.setGround() 


# In[15]:


#Calculate GCR
#gcr = CW / rtr
gcr = module.sceney / pitch 
gcr


# In[16]:


# -- establish tracking angles
trackerParams = {'limit_angle':50,
                 'angledelta':5,
                 'backtrack':True,
                 'gcr':gcr,
                 'cumulativesky':False,
                 'azimuth': 180,
                 }


# In[17]:


trackerdict = barn.set1axis(**trackerParams)
trackerdict = barn.gendaylit1axis()


# In[18]:


sceneDict = {'pitch':pitch, 
             'hub_height': hub_height,
             'nMods': 20,
             'nRows': 10}


# In[19]:


trackerdict = barn.makeScene1axis(module=module,sceneDict=sceneDict)


# In[20]:


trackerdict = barn.makeOct1axis()


# In[21]:


#!rvu -vf views\front.vp -e .01 -pe 0.4 -vp 0 -20 50 -vd 0 0.7 -0.7 1axis_2023-08-29_0900.oct


# In[23]:


# Modify modscanfront for Ground
resolutionGround = 1.2
modscanback = {'xstart': 0.4, 
                'zstart': 0.05,
                'xinc': resolutionGround,
                'zinc': 0,
                'Ny':5,
                'orient':'0 0 -1'}


# In[24]:


# counts from bottom right ...
trackerdict = barn.analysis1axis(trackerdict, customname = 'Ground', modWanted = 6, rowWanted = 5, modscanback=modscanback,
                                    sensorsy=1)


# In[25]:


filesall = os.listdir('results')
filestoclean = [e for e in filesall if e.endswith('_Front.csv')]
for cc in range(0, len(filestoclean)):
    filetoclean = filestoclean[cc]
    os.remove(os.path.join('results', filetoclean))


# In[ ]:


trackerdict = barn.analysis1axis(trackerdict, customname = 'Module', sensorsy=9)


# In[22]:


# compiling stuff, broken on this python version


# In[ ]:


ResultPVWm2Back = radObj.CompiledResults.iloc[0]['Grear_mean']
ResultPVWm2Front = radObj.CompiledResults.iloc[0]['Gfront_mean']


# In[ ]:


mykey = list(barn.trackerdict.keys())[0]
barn.trackerdict[mykey]['Results'][0]['AnalysisObj'].Wm2Back


# In[ ]:


#trackerdict = barn.calculateResults(bifacialityfactor=0.7, agriPV=True)


# In[ ]:


ghi_sum = metdata.ghi.sum()
ghi_sum
