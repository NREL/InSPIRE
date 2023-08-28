#!/usr/bin/env python
# coding: utf-8

# # 11 - AgriPV Systems
# 
# This journal shows how to model an AgriPV site, calculating the irradiance not only on the modules but also the irradiance received by the ground to evaluate available solar ersource for plants. 
# 
# We assume that bifacia_radiacne is already installed in your computer. This works for bifacial_radiance v.3 release.
# 
# These journal outlines 4 useful uses of bifacial_radiance and some tricks: 
# 
# * Creating the modules in the AgriPV site
# * Adding extra geometry for the pillars/posts supporting the AgriPV site
# * Hacking the sensors to sample the ground irradiance and create irradiance map
# * Adding object to simulate variations in ground albedo from different crops between rows.
# 
# 
# #### Steps:
# 
# 1. <a href='#step1'> Generate the geometry </a>
# 2. <a href='#step2'> Analyse the Ground Irradiance </a>
# 3. <a href='#step3'> Analyse and MAP the Ground Irradiance </a>
# 4. <a href='#step4'> Adding different Albedo Section </a>
#     
# #### Preview of what we will create: 
#     
# ![Another view](../images_wiki/AdvancedJournals/AgriPV_2.PNG)
# ![AgriPV Image We will create](../images_wiki/AdvancedJournals/AgriPV_1.PNG)
# And this is how it will look like:
# 
# ![AgriPV modeled step 4](../images_wiki/AdvancedJournals/AgriPV_step4.PNG)
# 
# 
# 

# <a id='step1'></a>

# ## 1. Generate the geometry 
# 
# This section goes from setting up variables to making the OCT axis. We are also adding some custom elements for the torquetubes and posts.
# 
# We've done this before a couple times, no new stuff here. 
# 
# The magic is that, for doing the carport we see in the figure, we are going to do a 4-up configuration of modules (**numpanels**), and we are going to repeat that 3-UP 6 times (**nMods**)

# In[1]:


import os
from pathlib import Path

testfolder = str(Path().resolve().parent / 'TEMP' /  'Tutorial_1')

if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)


# In[2]:


import bifacial_radiance as br
import numpy as np
import pandas as pd


# In[3]:


simulationname = 'tutorial_1'

#Location:
lat = 40.0583  # NJ
lon = -74.4057  # NJ

# MakeModule Parameters
moduletype='test-module'
numpanels = 3  # AgriPV site has 3 modules along the y direction (N-S since we are facing it to the south) .
x = 1  
y = 2
xgap = 2.0# Leaving 15 centimeters between modules on x direction
ygap = 0.10 # Leaving 10 centimeters between modules on y direction
zgap = 0 # no gap to torquetube.
sensorsy = 6*numpanels  # this will give 6 sensors per module, 1 per cell

# Other default values:

# TorqueTube Parameters
axisofrotationTorqueTube=False  # this is False by default if there is no torquetbue parameters
torqueTube = False
cellLevelModule = True

numcellsx = 12
numcellsy = 6
xcell = 0.156
ycell = 0.156
xcellgap = 0.02
ycellgap = 0.02

cellLevelModuleParams = {'numcellsx': numcellsx, 'numcellsy':numcellsy, 
                         'xcell': xcell, 'ycell': ycell, 'xcellgap': xcellgap, 'ycellgap': ycellgap}

# SceneDict Parameters
pitch = 15 # m
albedo = 0.2  #'grass'     # ground albedo
hub_height = 4.3 # m  
nMods = 6 # six modules per row.
nRows = 3  # 3 row

azimuth_ang=180 # Facing south
tilt =35 # tilt.


# In[5]:


demo = br.RadianceObj(simulationname,path = testfolder)  
demo.setGround(albedo) 
epwfile = demo.getEPW(lat, lon) # NJ lat/lon 40.0583° N, 74.4057
# Making module with all the variables
module=demo.makeModule(name=moduletype,x=x,y=y,numpanels=numpanels, 
                           xgap=xgap, ygap=ygap, cellModule=cellLevelModuleParams)
# create a scene with all the variables


# In[6]:


#startime and endtime 
#Valid options: mm_dd, mm_dd_HH, mm_dd_HHMM, YYYY-mm-dd_HHMM
metdata = demo.readWeatherFile(epwfile, coerce_year=2021, starttime='2021-06-17_1300', endtime='2021-06-17_1500')


# In[10]:


metdata.__dict__


# In[17]:


timeindex = metdata.datetime.index(pd.to_datetime('2021-06-17 13:0:0 -5'))  # Make this timezone aware, use -5 for EST.
timeindex


# In[18]:


demo.gendaylit(timeindex=timeindex)  


# In[20]:


sceneDict = {'tilt':tilt,'pitch': 15,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods, 'nRows': nRows}  
scene = demo.makeScene(module=moduletype, sceneDict=sceneDict) 
octfile = demo.makeOct(demo.getfilelist())


# If desired, you can view the Oct file at this point:
# 
# ***rvu -vf views\front.vp -e .01 tutorial_1.oct***

# In[21]:


## Comment the ! line below to run rvu from the Jupyter notebook instead of your terminal.
## Simulation will stop until you close the rvu window

get_ipython().system('rvu -vf views\\front.vp -e .01 tutorial_1.oct')


# And adjust the view parameters, you should see this image.
# 
# ![AgriPV modeled step 1](../images_wiki/AdvancedJournals/AgriPV_step1.PNG)
# 

# ### Adding a torquetube structure
# 
# We will add on the torquetube and pillars.
# 
# Positions of the piles could be done more programatically, but they are kinda estimated at the moment. 

# In[ ]:


module=demo.makeModule(name=moduletype,x=x,y=y,numpanels=numpanels, 
                           xgap=xgap, ygap=ygap, cellModule=cellLevelModuleParams, 
                      torquetube=True, )
# create a scene with all the variables


# ## 2. Analyse  the Ground Irradiance
# 
# Now let's do some analysis along the ground, starting from the edge of the modules. We wil select to start in the center of the array.
# 
# We are also increasign the number of points sampled accross the collector width, with the  variable **sensorsy** passed to **moduleanalysis**. We are also increasing the step between sampling points, to be able to sample in between the rows.

# In[25]:


analysis = br.AnalysisObj(octfile, demo.name)  
sensorsy = 20
frontscan, backscan = analysis.moduleAnalysis(scene, sensorsy=sensorsy)


# In[26]:


groundscan = frontscan


# In[27]:


groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
groundscan['zinc'] = 0   # no tilt necessary. 
groundscan['yinc'] = pitch/(sensorsy-1)   # increasing spacing so it covers all distance between rows
groundscan


# In[28]:


analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan)  # compare the back vs front irradiance  


# This is the result for only one 'chord' accross the ground. Let's now do a X-Y scan of the ground.

# <a id='step3'></a>

# ## 3. Analyse and MAP the Ground Irradiance

#  We will use the same technique to find the irradiance on the ground used above, but will move it along the X-axis to map from the start of one module to the next.
#  
#  We will sample around the module that is placed at the center of the field.

# ![AgriPV modeled step 4](../images_wiki/AdvancedJournals/spacing_between_modules.PNG)

# In[29]:


import seaborn as sns


# In[30]:


sensorsx = 20
startgroundsample=-module.scenex
spacingbetweensamples = module.scenex/(sensorsx-1)

for i in range (0, sensorsx): # Will map 20 points    
    frontscan, backscan = analysis.moduleAnalysis(scene, sensorsy=sensorsy)
    groundscan = frontscan
    groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
    groundscan['zinc'] = 0   # no tilt necessary. 
    groundscan['yinc'] = pitch/(sensorsy-1)   # increasing spacing so it covers all distance between rows
    groundscan['xstart'] = startgroundsample + i*spacingbetweensamples   # increasing spacing so it covers all distance between rows
    analysis.analysis(octfile, simulationname+"_groundscan_"+str(i), groundscan, backscan)  # compare the back vs front irradiance  


# Read all the files generated into one dataframe

# In[31]:


filestarter = "irr_AgriPV_groundscan_"

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
        resultsDF = load.read1Result(os.path.join(testfolder, 'results', prefixed[i]))
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
