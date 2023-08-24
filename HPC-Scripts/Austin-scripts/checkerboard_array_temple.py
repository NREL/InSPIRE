#!/usr/bin/env python
# coding: utf-8

# Checkerboard array - Temple Ambler Campus
# by Austin Kinzer

# NOTE: 
# This script is derived from the bifacial_radiance tutorial on east-west sheds
# Variable names may be holdovers from Tutorial 19


# In[1]:


import os
import numpy as np
import pandas as pd
from pathlib import Path
import bifacial_radiance


# In[2]:


bifacial_radiance.__version__


# In[3]:

basefolder = os.path.join(os.getcwd(), 'TEMP')

#testfolder = testfolder = str(Path().resolve().parent.parent / 'bifacial_radiance' / 'Tutorial_01')
#if not os.path.exists(testfolder):
#    os.makedirs(testfolder)

demo = bifacial_radiance.RadianceObj("tutorial_19", path = basefolder)  # Create a RadianceObj 'object'
demo.setGround(0.2) # albedo
simulationname = "Temple_checkerboard"


lat = 40.16492110325204  # Ambler, PA
lon = -75.19263121534482 # Ambler, PA

epwfile = demo.getEPW(lat = lat, lon = lon)    
metdata = demo.readWeatherFile(epwfile, coerce_year=2001) 
demo.genCumSky()


# Define your shed characteristics. In this case it is a 1-up portrait setup:

# In[4]:

# 79.92 × 39.53 × 1.38 in --- panel dimensions

# For sanity check, we are creating the same module but with different names for each orientation.
numpanels=1 
ygap = 0.0 # m Spacing between modules on each shed. not relevant for 1-up. Typically value is ~0.02
y=2.030  # m. module size, one side
x=1.004 # m. module size, other side. for landscape, x > y
mymoduleA = demo.makeModule(name='test-module_A',y=y,x=x, numpanels=numpanels, ygap=ygap) # create module A
mymoduleB = demo.makeModule(name='test-module_B',y=y,x=x, numpanels=numpanels, ygap=ygap) # create module B


# In[5]:


tilt = 5 # degrees
gap_between_EW_sheds = 0.30 # m --- row gap 1
gap_between_shed_rows = 0.30 #m --- row gap 2 - should be same for checkerboard. This notation is a remnant of the E-W racking
xgap = x # set xgap equal to panel length
CW = mymoduleA.sceney
ground_underneat_shed = CW * np.cos(np.radians(tilt)) # horizontal distance of panel (calculated based on tilt angle and panel length)
pitch = ground_underneat_shed*2 + gap_between_EW_sheds + gap_between_shed_rows # 
offset_westshed = -(ground_underneat_shed+gap_between_EW_sheds) # set distance between panels along row
# offset_rowsB = x


# Define the other characteristics of our array:

# In[6]:


clearance_height = 3 # m
nMods = 4
nRows = 2


# Create the Scene Objects and the Scene:

# In[7]:


sceneDict = {'tilt':tilt,'pitch':pitch,'clearance_height':clearance_height,'azimuth':0, 'xgap': xgap, 'nMods': nMods, 'nRows': nRows, 
             'appendRadfile':True} 
sceneObj1 = demo.makeScene(mymoduleA, sceneDict)  

sceneDict2 = {'tilt':tilt,'pitch':pitch,'clearance_height':clearance_height,'azimuth':0, 'xgap': xgap, 'nMods': (nMods - 1), 'nRows': nRows, 
              'originx': offset_westshed, 'originy': pitch*0.5, 
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
# We have to analyze the East and the West shed independently. 

# In[11]:


sensorsy=4  # 1 per module. consider increasing the number but be careful with sensors in the space between modules.
analysis = bifacial_radiance.AnalysisObj(octfile, demo.basename)  
frontscan, backscan = analysis.moduleAnalysis(sceneObj1, sensorsy=sensorsy)
frontdict, backdict = analysis.analysis(octfile, "EastFacingShed", frontscan, backscan)  # compare the back vs front irradiance  

frontscan, backscan = analysis.moduleAnalysis(sceneObj2, sensorsy=sensorsy )
frontdict2, backdict2 = analysis.analysis(octfile, "WestFacingShed", frontscan, backscan)  # compare the back vs front irradiance  






# In[12]:
    
# ## Analysis

sensorsy=3  # 1 per module. consider increasing the number but be careful with sensors in the space between modules.
analysis = bifacial_radiance.AnalysisObj(octfile, demo.basename)  
#frontscan, backscan = analysis.moduleAnalysis(sceneObj1, sensorsy=sensorsy)
#frontdict, backdict = analysis.analysis(octfile, "Row A", frontscan, backscan)  # compare the back vs front irradiance  

#frontscan, backscan = analysis.moduleAnalysis(sceneObj2, sensorsy=sensorsy )
#frontdict2, backdict2 = analysis.analysis(octfile, "Row B", frontscan, backscan)  # compare the back vs front irradiance  

# ground analysis

sensorsy = 21
# frontscan, backscan = analysis.moduleAnalysis(sceneObj1, sensorsy=sensorsy)
# groundscan = frontscan
# groundscan['zstart'] = 0.05
# groundscan['zinc'] = 0
# groundscan['yinc'] = pitch/2/(sensorsy-1)
# groundscan

# analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan)

sensorsx = 21
startgroundsample = -mymoduleA.scenex
# startgroundsample_y = -mymoduleA.sceney
# spacingbetweensamples = mymoduleA.scenex/(sensorsx - 1)
spacingbetweensamples_x = 1
spacingbetweensamples_y = 1

for i in range (0, sensorsx): # Will map x points
    frontscan, backscan = analysis.moduleAnalysis(sceneObj1, sensorsy=sensorsy)
    groundscan = frontscan
    groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
    groundscan['ystart'] = -10 # 
    groundscan['zinc'] = 0   # no tilt necessary.
    groundscan['yinc'] = spacingbetweensamples_y   # increasing spacing so it covers all distance between rows
    groundscan['xstart'] = -10 + i*spacingbetweensamples_x   # increasing spacing so it covers all distance between rows
    analysis.analysis(octfile, simulationname+"_groundscan_"+str(i), groundscan, backscan)  # compare the back vs front irradiance

    
print("FINISHED GROUND ANALYSIS")
