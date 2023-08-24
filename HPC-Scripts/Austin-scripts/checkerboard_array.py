#!/usr/bin/env python
# coding: utf-8

# Checkerboard array - Kebwezi, Kenya
# 

# In[1]:


import os
import numpy as np
import pandas as pd
from pathlib import Path
import bifacial_radiance
import seaborn as sns


# In[2]:


bifacial_radiance.__version__


# In[3]:


testfolder = str(Path().resolve().parent.parent / 'bifacial_radiance' / 'Tutorial_01')
if not os.path.exists(testfolder):
    os.makedirs(testfolder)

demo = bifacial_radiance.RadianceObj("tutorial_19", path = testfolder)  # Create a RadianceObj 'object'
demo.setGround(0.2)

simulationname = "Kebwezi_checkerboard"

lat = -2.3247242925313114  # Kibwezi, Kenya
lon = 37.97558795280663 # Kibwezi, Kenya

epwfile = demo.getEPW(lat = lat, lon = lon)    
metdata = demo.readWeatherFile(epwfile, coerce_year=2001) 
demo.genCumSky()


# Define your shed characteristics. In this case it is a 4-up landscape setup:

# In[4]:


# For sanity check, we are creating the same module but with different names for each orientation.
numpanels=3 
ygap = 0.02 # m Spacing between modules on each shed.
y=1.096  # m. module size, one side
x=2.384 # m. module size, other side. for landscape, x > y
mymoduleA = demo.makeModule(name='test-module_A',y=y,x=x, numpanels=numpanels, ygap=ygap)
mymoduleB = demo.makeModule(name='test-module_B',y=y,x=x, numpanels=numpanels, ygap=ygap)


# Calculate the spacings so we can offset the West Facing modules properly:
# 
# ![East West Sheds Example](../images_wiki/AdvancedJournals/EW_sheds_Offset.PNG)
# 
# 

# In[5]:


tilt = 1 # degrees
gap_between_EW_sheds = 0.10 # m
gap_between_shed_rows = 0.10 #m
xgap = y # set xgap equal to panel length
CW = mymoduleA.sceney
ground_underneat_shed = CW * np.cos(np.radians(tilt))
pitch = ground_underneat_shed*2 + gap_between_EW_sheds + gap_between_shed_rows
offset_westshed = -(ground_underneat_shed+gap_between_EW_sheds)
offset_rowsB = x


# Define the other characteristics of our array:

# In[6]:


clearance_height = 3 # m
nMods = 6
nRows = 6


# Create the Scene Objects and the Scene:

# In[7]:


sceneDict = {'tilt':tilt,'pitch':pitch,'clearance_height':clearance_height,'azimuth':0, 'xgap': xgap, 'nMods': nMods, 'nRows': nRows, 
             'appendRadfile':True} 
sceneObj1 = demo.makeScene(mymoduleA, sceneDict)  

sceneDict2 = {'tilt':tilt,'pitch':pitch,'clearance_height':clearance_height,'azimuth':0, 'xgap': xgap, 'nMods': nMods, 'nRows': nRows, 
              'originx': offset_westshed, 'originy': offset_rowsB, 
              'appendRadfile':True} 

sceneObj2 = demo.makeScene(mymoduleB, sceneDict2)  


# Finally get all the files together by creating the Octfile:

# In[8]:


octfile = demo.makeOct(demo.getfilelist()) 


# ## View the Geometry
# 
# You can check the geometry on rvu with the following commands. You can run it in jupyter/Python if you comment the line, but the program will not continue processing until you close the rvu window. ( if running rvu directly on the console, navigate to the folder where you have the simulation, and don't use the exclamation point at the beginning)
# 
# Top view:

# In[9]:


#!rvu -vf views\front.vp -e .01 -pe 0.3 -vp 1 -45 40 -vd 0 0.7 -0.7 MultipleObj.oct


# another view, close up:

# In[10]:


# !rvu -vf views\front.vp -e .01 -pe 0.3 -vp -4 -29 3.5 -vd 0 1 0 MultipleObj.oct


# ## Analysis
# 

# In[11]:


sensorsy=3  # 1 per module. consider increasing the number but be careful with sensors in the space between modules.
analysis = bifacial_radiance.AnalysisObj(octfile, demo.basename)  
frontscan, backscan = analysis.moduleAnalysis(sceneObj1, sensorsy=sensorsy)
frontdict, backdict = analysis.analysis(octfile, "Row A", frontscan, backscan)  # compare the back vs front irradiance  

frontscan, backscan = analysis.moduleAnalysis(sceneObj2, sensorsy=sensorsy )
frontdict2, backdict2 = analysis.analysis(octfile, "Row B", frontscan, backscan)  # compare the back vs front irradiance  

# ground analysis

sensorsy = 48
# frontscan, backscan = analysis.moduleAnalysis(sceneObj1, sensorsy=sensorsy)
# groundscan = frontscan
# groundscan['zstart'] = 0.05
# groundscan['zinc'] = 0
# groundscan['yinc'] = pitch/2/(sensorsy-1)
# groundscan

# analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan)

sensorsx = 48
startgroundsample = -mymoduleA.scenex
spacingbetweensamples = mymoduleA.scenex/(sensorsx - 1)

for i in range (0, sensorsx): # Will map 20 points
    frontscan, backscan = analysis.moduleAnalysis(sceneObj1, sensorsy=sensorsy)
    groundscan = frontscan
    groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
    groundscan['zinc'] = 0   # no tilt necessary.
    groundscan['yinc'] = pitch/(sensorsy-1)   # increasing spacing so it covers all distance between rows
    groundscan['xstart'] = startgroundsample + i*spacingbetweensamples   # increasing spacing so it covers all distance between rows
    analysis.analysis(octfile, simulationname+"_groundscan_"+str(i), groundscan, backscan)  # compare the back vs front irradiance
    
