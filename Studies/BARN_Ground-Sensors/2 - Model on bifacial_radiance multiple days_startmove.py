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

# In[2]:


import os
from pathlib import Path

# testfolder = Path().resolve().parent.parent / 'TEMP' / 'bifacial_radiance'

testfolder = os.path.join('TEMP', 'BARN_valid_inchwest_feb2_2023')

if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)


# In[3]:


print(testfolder)


# In[4]:


import bifacial_radiance as br
import numpy as np
import pandas as pd


# In[5]:


# This information helps with debugging and getting support :)
import sys, platform
print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("bifacial_radiance version ", br.__version__)


# In[6]:


simulationname = 'BARN_br_validation_inchwest'

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


# In[8]:


weatherfile = str(Path().resolve().parent.parent / 'WeatherFiles' /  'PSM3_15T.csv')
print(weatherfile)


# In[20]:


#starttime='2021-08-30_09', endtime='2021-08-30_09'
#Valid options: mm_dd, mm_dd_HH, mm_dd_HHMM, YYYY-mm-dd_HHMM
#metdata = barn.readWeatherFile(weatherfile, coerce_year=2023, source='SAM', starttime='2023-08-29_09', endtime='2023-08-29_09', label='right')
# Gather weather data from 9-20-2023 to 9-29-2023
metdata = barn.readWeatherFile(weatherfile, coerce_year=2023, source='SAM', starttime='2023-09-20', endtime='2023-09-20', label='center')


# In[21]:


module=barn.makeModule(name=moduletype,x=module_x,y=module_y)


# In[22]:


barn.metdata.__dict__


# In[23]:


barn.setGround() 


# In[24]:


#Calculate GCR
#gcr = CW / rtr
gcr = module.sceney / pitch 
gcr


# In[25]:


# -- establish tracking angles
trackerParams = {'limit_angle':50,
                 'angledelta':5,
                 'backtrack':True,
                 'gcr':gcr,
                 'cumulativesky':False,
                 'azimuth': 180,
                 }


# In[26]:


trackerdict = barn.set1axis(**trackerParams)
trackerdict = barn.gendaylit1axis()


# In[27]:


sceneDict = {'pitch':pitch, 
             'hub_height': hub_height,
             'nMods': 20,
             'nRows': 10}


# In[28]:


trackerdict = barn.makeScene1axis(module=module,sceneDict=sceneDict)
# trackerdict = barn.loadtrackerdict()
# trackerdict = barn.returnOctFiles()


# In[29]:


trackerdict = barn.makeOct1axis()


# In[19]:


# !rvu -vf views\front.vp -e .01 -pe 0.4 -vp 0 -20 50 -vd 0 0.7 -0.7 1axis_2023-09-21_1200.oct


# In[ ]:


offsets = [-2, -1, 0, 1, 2]
inchtom = 0.0254

sensorsposition_x = [0.4254, 1.6254, 2.8254, 4.0254, 5.2254]


# In[1]:


for ii in range(0, len(offsets)):
    for sensor in range(0, len(sensorsposition_x)):
        offset = offsets[ii]*inchtom  # passing to meters.
        xstart = sensorsposition_x[sensor]
        
        modscanback = {'xstart': xstart+offset, # an inch east of 0.4 meter starting point
                        'zstart': 0.05,
                        'xinc': 0,
                        'zinc': 0,
                        'Ny':1,
                        'orient':'0 0 -1'}
        # counts from bottom right ...
        trackerdict = barn.analysis1axis(trackerdict, customname = 'Ground_S'+str(sensor+1)+'_offset_'+str(offsets[ii])+'in', 
                                         modWanted = 7, rowWanted = 4, 
                                         modscanfront=modscanback, sensorsy=1)


# In[ ]:


# remove unwanted results, in this case all the "fronts"
filesall = os.listdir('results')
filestoclean = [e for e in filesall if e.endswith('_Front.csv')]
for cc in range(0, len(filestoclean)):
    filetoclean = filestoclean[cc]
    os.remove(os.path.join('results', filetoclean))


# In[ ]:


from datetime import datetime
import re


# In[ ]:


filestarter = "irr_1axis_2023-09-16_"
filelist = sorted(os.listdir(os.path.join(testfolder, 'results')))
prefixed = [filename for filename in filelist if filename.startswith(filestarter)]
# print(prefixed)
distance = []
groundirrad = []
filenamed = []
faillist = []
Datetime = []
Ap_1 = []
Ap_2 = []
Ap_3 = []
Ap_4 = []
Ap_5 = []

print('{} files in the directory'.format(filelist.__len__()))
print('{} groundscan files in the directory'.format(prefixed.__len__()))
i = 0  # counter to track # files loaded.

# for i in range (0, len(prefixed)):
#     ind = prefixed[i].split('_')

#     try:
# #         Datetime.append(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}\_[0-9]{4})", prefixed[i])[0], '%Y-%m-%d_%H%M')
#         resultsDF = br.load.read1Result(os.path.join(testfolder, 'results', prefixed[i]))
# #         distance.append(list(resultsDF['x']))
# #         groundirrad.append(list(resultsDF['Wm2Back']))
#         distance.append(list(resultsDF['x']))
#         groundirrad.append(list(resultsDF['Wm2Back']))

#         filenamed.append(prefixed[i])
#     except:
#         print(" FAILED ", i, prefixed[i])
#         faillist.append(prefixed[i])

for i in range (0, len(prefixed)):
    ind = prefixed[i].split('_')

    try:
        Datetime.append(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}\_[0-9]{4})", prefixed[i])[0])
        resultsDF = br.load.read1Result(os.path.join(testfolder, 'results', prefixed[i]))
#         ap1.append(list(resultsDF['Wm2Back'][[0]]))
#         ap2.append(list(resultsDF['Wm2Back'][[1]]))
#         ap3.append(list(resultsDF['Wm2Back'][[2]]))
#         ap4.append(list(resultsDF['Wm2Back'][[3]]))
#         ap5.append(list(resultsDF['Wm2Back'][[4]]))
        Ap_1.append(resultsDF['Wm2Back'][0])
        Ap_2.append(resultsDF['Wm2Back'][1])
        Ap_3.append(resultsDF['Wm2Back'][2])
        Ap_4.append(resultsDF['Wm2Back'][3])
        Ap_5.append(resultsDF['Wm2Back'][4])

        filenamed.append(prefixed[i])
    except:
        print(" FAILED ", i, prefixed[i])
        faillist.append(prefixed[i])
        
# resultsdf = pd.DataFrame(list(zip(distance, groundirrad)), columns = ['br_position', 'br_irradiance'])
resultsdf = pd.DataFrame(list(zip(Ap_1, Ap_2, Ap_3, Ap_4, Ap_5)), columns = ['Ap_1', 'Ap_2', 'Ap_3', 'Ap_4', 'Ap_5'])

resultsdf['Datetime'] = Datetime
resultsdf['Datetime'] = pd.to_datetime(resultsdf['Datetime'], format ='%Y-%m-%d_%H%M')


# print(resultsdf)


# In[ ]:


resultsdf.to_csv('out6.csv',index=False)


# In[ ]:


# Tidy the data
resultsdf_melt = resultsdf.melt(id_vars = ['Datetime'], var_name = 'position', value_name = 'value')
resultsdf_melt['datatype'] = 'modeled'
print(resultsdf_melt['Datetime'])

# Load the measured data
measured = pd.read_csv(os.path.join(os.path.join(Path().resolve().parent.parent, 'Data','BARNirrad_measured.csv')), header = 1)
# print(resultsdf_melt.info)
# measured.info

measured['Datetime'] = pd.to_datetime(measured['TIMESTAMP'], format ='%m/%d/%Y %H:%M')
# measured_select = measured[['Datetime', 'Ap_1', 'Ap_2', 'Ap_3', 'Ap_4', 'Ap_5']]
measured_melt = measured[['Datetime', 'Ap_1', 'Ap_2', 'Ap_3', 'Ap_4', 'Ap_5']].melt(
    id_vars = ['Datetime'],
    var_name = 'position', 
    value_name = 'value')
measured_melt['datatype'] = 'measured'

# print(measured_melt)

combined = pd.merge(resultsdf_melt, measured_melt, how = 'left', on = 'Datetime')
measured_melt = pd.merge(resultsdf_melt[['Datetime']], measured_melt, how = 'left', on = 'Datetime')
measured_melt.drop_duplicates(inplace=True)
print(measured_melt)
combined = pd.concat([resultsdf_melt, measured_melt])
# combined.to_csv('combined.csv', index = False)
print(combined)


# In[ ]:


import plotly.express as px


# In[ ]:


fig = px.line(combined, x = 'Datetime', y = 'value', color = 'datatype', facet_col = 'position', facet_col_wrap = 1)
fig.show()


# In[ ]:



