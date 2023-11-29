#!/usr/bin/env python
# coding: utf-8

# <a id='step1'></a>

# In[1]:


import os
from pathlib import Path

testfolder = str(Path().resolve().parent / 'TEMP' /  'MA')

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
lat = 42.3732  # MA
lon = -72.5199  # MA

# Scene Parameters:
azimuth_ang=180 # Facing south
tilt =35 # tilt.
pitch = 15 # m 
albedo = 0.2  #'grass'     # ground albedo
hub_height = 4.3 # m  
nRows = 2

# MakeModule Parameters
moduletype='pv-collector'
numpanels = 3  # AgriPV site has 3 modules along the y direction
x = 2 # m
y = 1 # m. slope we will measure
# if x > y, landscape. if x < y, portrait
sensorsy = 6*numpanels  # this will give 6 sensors per module, 1 per cell

ft2m = 0.3048

# Scene Object 1
xcoord_1 = 0
ycoord_1 = 0
nMods_1 = 3
xgap_1 = 2*ft2m # 2f converted to meters now

# Scene Object 2
nMods_2 = 2
xgap_2 = 3.5*ft2m # ft
ycoord_2 = 0 
xcoord_2 = x*nMods_1/2+xgap_1 + xgap_2 + x/2

# Scene Object 3
nMods_3 = 2
xgap_3 = 5*ft2m # ft
ycord_3 = 0
xcoord_3 = x*nMods_1/2+xgap_1+ xgap_2 + x*2 + xgap_2+xgap_3 + x/2

# TorqueTube Parameters
tubetype='square' # Other options: 'square' , 'hex'
material = 'Metal_Grey' # Other options: 'black'
diameter = 0.1 # m
axisofrotationTorqueTube = False
zgap = 0.05 # m
visible = True 

#Add torquetube 
tubeParams = {'tubetype':tubetype,
              'diameter':diameter,
              'material':material,
              'axisofrotation':False,
              'visible':True}


# In[5]:


NREL_API_KEY = None  # <-- please set your NREL API key here
# note you must use "quotes" around your key as it is a string.

if NREL_API_KEY is None:
       NREL_API_KEY = 'DEMO_KEY'  # OK for this demo, but better to get your own key

import pvlib

metdata, metadata = pvlib.iotools.get_psm3(
    latitude=lat, longitude=lon,
    api_key=NREL_API_KEY,
    email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
    names='tmy', map_variables=True)

# Some of the names changed internally. While bifacial_radiance updates their expected names, we are renaming the values here
metadata['timezone'] = metadata['Time Zone']
metadata['county'] = '-'
metadata['elevation'] = metadata['altitude']
metadata['state'] = metadata['State']
metadata['country'] = metadata['Country']
metdata['Albedo'] = metdata['albedo']


# In[6]:


demo = br.RadianceObj(simulationname,path = testfolder)  
demo.setGround(albedo) 


# In[7]:


# Specifiying growth season May to Oct.
metData = demo.NSRDBWeatherData(metadata, metdata, starttime='05_09', endtime='10_01',coerce_year=2021)


# In[8]:


module_1=demo.makeModule(name='mod1',x=x,y=y,numpanels=numpanels, 
                           xgap=xgap_1, tubeParams=tubeParams)

module_2=demo.makeModule(name='mod2',x=x,y=y,numpanels=numpanels, 
                           xgap=xgap_2, tubeParams=tubeParams)

module_3=demo.makeModule(name='mod3',x=x,y=y,numpanels=numpanels, 
                           xgap=xgap_3, tubeParams=tubeParams)


# In[ ]:





# In[9]:


demo.genCumSky()  


# In[10]:


#timeindex = metdata.datetime.index(pd.to_datetime('2021-06-21 12:0:0 -5'))  # Make this timezone aware, use -5 for EST.
#demo.gendaylit(timeindex=timeindex)  


# In[11]:


sceneDict_1 = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods_1, 'nRows': nRows}
sceneDict_2 = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods_2, 'nRows': nRows, 
                'originx': xcoord_2, 'appendRadfile':True} 
sceneDict_3 = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods_3, 'nRows': nRows, 
               'originx': xcoord_3, 'appendRadfile':True}  

scene_1 = demo.makeScene(module=module_1, sceneDict=sceneDict_1) 
scene_2 = demo.makeScene(module=module_2, sceneDict=sceneDict_2) 
scene_3 = demo.makeScene(module=module_3, sceneDict=sceneDict_3) 



# In[12]:


demo.getfilelist()


# In[ ]:


#demo.gendaylit(timeindex=5)  


# In[13]:


octfile = demo.makeOct()


# In[ ]:


## Comment the ! line below to run rvu from the Jupyter notebook instead of your terminal.
## Simulation will stop until you close the rvu window

#!rvu -vf views\front.vp -e .01 tutorial_1.oct


# In[26]:


analysis = br.AnalysisObj(octfile, demo.name)  
frontscan, backscan = analysis.moduleAnalysis(scene_1, sensorsy=sensorsy, sensorsx=4, modWanted=3)


# In[27]:


groundscan = frontscan
groundscan


# In[28]:


groundscan['xstart'] = -4  
groundscan['ystart'] = -1.0  
groundscan['zstart'] = 0.05  
groundscan['xinc'] = 0    
groundscan['yinc'] = 1  
groundscan['zinc'] = 0   
groundscan['sx_xinc'] = 2   # here's hte trick. this moves the xstart once every loop.
groundscan['sy_xinc'] = 0 
groundscan['sz_xinc'] = 0   
groundscan['Nx'] = 10   
groundscan['Ny'] = 3  
groundscan['Nz'] = 1  
groundscan['orient'] = '0 0 -1' 
groundscan


# In[41]:


results_ground, results_ignore = analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan)  # compare the back vs front irradiance  


# In[47]:


import seaborn as sns


# Read all the files generated into one dataframe

# In[ ]:


filestarter = "irr_tutorial_1_groundscan_"

filelist = sorted(os.listdir(os.path.join(testfolder, 'results')))
prefixed = [filename for filename in filelist if filename.startswith(filestarter)]
arrayWm2Front = []
arrayWm2Back = []
arrayMatFront = []
arrayMatBack = []
filenamed = []
faillist = []

print('{} files in the directory'.format(filelist.__len__()))
print('{} groundscan files in the directory'.format(prefixed.__len__()))
i = 0  # counter to track # files loaded.

for i in range (0, len(prefixed)):
    ind = prefixed[i].split('_')

    try:
        resultsDF = br.load.read1Result(os.path.join(testfolder, 'results', prefixed[i]))
        arrayWm2Front.append(list(resultsDF['Wm2Front']))
        arrayWm2Back.append(list(resultsDF['Wm2Back']))
        arrayMatFront.append(list(resultsDF['mattype']))
        arrayMatBack.append(list(resultsDF['rearMat']))
        filenamed.append(prefixed[i])
    except:
        print(" FAILED ", i, prefixed[i])
        faillist.append(prefixed[i])

resultsdf = pd.DataFrame(list(zip(arrayWm2Front, arrayWm2Back, 
                                  arrayMatFront, arrayMatBack)),
                         columns = ['br_Wm2Front', 'br_Wm2Back', 
                                    'br_MatFront', 'br_MatBack'])
resultsdf['filename'] = filenamed


# Creating a new dataframe where  each element in the front irradiance list is a column. Also transpose and reverse so it looks like a top-down view of the ground.

# In[ ]:


df3 = pd.DataFrame(resultsdf['br_Wm2Front'].to_list())
reversed_df = df3.T.iloc[::-1]


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[ ]:


# Plot
ax = sns.heatmap(reversed_df)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('')  
ax.set_xlabel('')  
print('')


# <a id='step4'></a>
