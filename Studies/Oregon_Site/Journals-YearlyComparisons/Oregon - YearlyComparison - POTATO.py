#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path

testfolder = os.path.join('TEMP','PotatoY1-Y3')

if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)


# In[2]:


from bifacial_radiance import *
import numpy as np
import pandas as pd
import datetime
import pvlib


# In[3]:


# This information helps with debugging and getting support :)
import sys, platform
import bifacial_radiance as br
print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("bifacial_radiance version ", br.__version__)
print("PVLib version ", pvlib.__version__)


# In[4]:


startdt = datetime.datetime(2001,3,15,0)
enddt = datetime.datetime(2001,6,30,23)


# In[5]:


NREL_API_KEY = None  # <-- please set your NREL API key here
# note you must use "quotes" around your key as it is a string.

if NREL_API_KEY is None:
       NREL_API_KEY = 'DEMO_KEY'  # OK for this demo, but better to get your own key


# In[6]:


#Site 1
#44.57615187732146,-123.23914850912513
lat_1=44.57615187732146
lon_1=-123.23914850912513
lat_2=44.566202648983094
lon_2=-123.30914089141844
#Site 2


# In[7]:


data2019_S1, metdata2019_S1 = pvlib.iotools.get_psm3(
    latitude=lat_1, longitude=lon_1,
    api_key=NREL_API_KEY,
    email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
    names='2019', map_variables=True, leap_day=False)

data2020_S1, metdata2020_S1 = pvlib.iotools.get_psm3(
    latitude=lat_1, longitude=lon_1,
    api_key=NREL_API_KEY,
    email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
    names='2020', map_variables=True, leap_day=False)

data2021_S1, metdata2021_S1 = pvlib.iotools.get_psm3(
    latitude=lat_1, longitude=lon_1,
    api_key=NREL_API_KEY,
    email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
    names='2021', map_variables=True, leap_day=False)


# In[8]:


data2019_S2, metdata2019_S2 = pvlib.iotools.get_psm3(
latitude=lat_2, longitude=lon_2,
api_key=NREL_API_KEY,
email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
names='2019', map_variables=True, leap_day=False)

data2020_S2, metdata2020_S2 = pvlib.iotools.get_psm3(
latitude=lat_2, longitude=lon_2,
api_key=NREL_API_KEY,
email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
names='2020', map_variables=True, leap_day=False)

data2021_S2, metdata2021_S2 = pvlib.iotools.get_psm3(
latitude=lat_2, longitude=lon_2,
api_key=NREL_API_KEY,
email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
names='2021', map_variables=True, leap_day=False)


# In[9]:


lat_site1 = 44.57615187732146   
lon_site2 = -123.23914850912513
clearance_heights = [0.88, 0.9482582, 0.6985] # m
ygaps = [0.02, 0.02, 0.02] # m
cws = [3.3655, 3.3655, 3.9624] # m
rtrs = [6.223, 8.4201, 6.8453] # m
tilt = 25
sazm = 180
albedo = 0.2 # 'grass'
years=[2019,2020,2021]
datasets_S2 = [data2019_S2, data2020_S2, data2021_S2]
metdataset_S2 = [metdata2019_S2, metdata2020_S2, metdata2021_S2]

datasets_S1 = [data2019_S1, data2020_S1, data2021_S1]
metdataset_S1 = [metdata2019_S1, metdata2020_S2, metdata2021_S1]

# Field size. Just going for 'steady state'
nMods = 20
nRows = 7


# In[10]:


startdts = [datetime.datetime(2001,4,1,0),
            datetime.datetime(2001,5,1,0),
            datetime.datetime(2001,6,1,0),
            datetime.datetime(2001,7,1,0),
            datetime.datetime(2001,8,1,0),
            datetime.datetime(2001,9,1,0),
            datetime.datetime(2001,10,1,0),
            datetime.datetime(2001,4,1,0)]

enddts = [datetime.datetime(2001,4,30,23),
          datetime.datetime(2001,5,31,23),
          datetime.datetime(2001,6,30,23),
          datetime.datetime(2001,7,31,23),
          datetime.datetime(2001,8,31,23),
          datetime.datetime(2001,9,30,23),
          datetime.datetime(2001,10,15,0),
          datetime.datetime(2001,10,15,0)]


# In[11]:


demo = RadianceObj('oregon', path=testfolder)
demo.setGround(0.2)


# In[13]:


simulate = True

if simulate:
    for setup in range(0, 2):
        for year in range(0, 3):
            for mmonth in range(0, len(startdts)):
                y = (cws[setup]-ygaps[setup])/2
                year_str = years[year]
                
                if setup==0:
                    weather = datasets_S1[year]
                    meta = metdataset_S1[year]
                else:
                    weather = datasets_S2[year]
                    meta = metdataset_S2[year]
                    
                module = demo.makeModule(name='module_site'+str(setup+1), x=1, y=y, numpanels=2, 
                                        ygap=ygaps[setup])
                startdt = startdts[mmonth]
                enddt = enddts[mmonth]
                metdata = demo.NSRDBWeatherData(meta, weather,starttime=startdt, endtime=enddt, coerce_year=2001) # read in the EPW weather data from above
                demo.genCumSky(savefile=str(mmonth))
                #demo.gendaylit(4020)  # Use this to simulate only one hour at a time. 

                sceneDict = {'tilt':tilt, 'pitch':rtrs[setup], 'clearance_height':clearance_heights[setup], 
                             'azimuth':sazm, 'nMods':nMods, 'nRows':nRows}  
                scene = demo.makeScene(module=module, sceneDict=sceneDict) 
                octfile = demo.makeOct(demo.getfilelist())  

                analysis = AnalysisObj(octfile, demo.name)
                
                # Module first
                frontscan, backscan = analysis.moduleAnalysis(scene, sensorsx = 1, sensorsy=10)
                analysis.analysis(octfile, 'MODULE_setup_'+(str(setup+1))+'_'+str(year_str)+'_'+str(startdt.month)+'to'+str(enddt.month)+'_', frontscan, backscan)  # compare the back vs front irradiance  

                # Ground
                # spacingbetweensamples = 0.05 # m
                # sensorsy = int(np.floor(rtrs[setup]/spacingbetweensamples)+1)
                sensorsx = 1
                ft2m=0.3556
                bedloc = 0.5*cws[setup]*np.cos(np.radians(tilt))+2.5*ft2m  # Edge + 2.5ft
                bedlocinc = 5*ft2m # 2nd bed is 5 feet from bed 1
                
                groundscan, backscan = analysis.moduleAnalysis(scene, sensorsx = 1, sensorsy=[2, 1])
                groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
                groundscan['zinc'] = 0   # no tilt necessary. 
                groundscan['ystart'] = bedloc
                groundscan['yinc'] = bedlocinc
                analysis.analysis(octfile, 'GROUND_setup_'+(str(setup+1))+'_'+str(year_str)+'_'+str(startdt.month)+'to'+str(enddt.month)+'_', groundscan, backscan)  # compare the back vs front irradiance  

    filesall = os.listdir('results')

    # Cleanup of Ground 'back' files
    filestoclean = [e for e in filesall if e.endswith('_Back.csv')]
    for cc in range(0, len(filestoclean)):
        filetoclean = filestoclean[cc]
        os.remove(os.path.join('results', filetoclean))

