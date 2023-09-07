#!/usr/bin/env python
# coding: utf-8

# # 1. Read DAT File
# Typical workflow: go to https://midcdmz.nrel.gov/apps/day.pl?BMS 
# 
# Get the following fields for ALL TIMES (not just sunrise):
# 
# - Global CMP22 (vent/cor) [W/m^2]
# - Direct CHP1-1 [W/m^2]
# - Diffuse 8-48 (vent) [W/m^2]
# - Tower Dry Bulb Temp [deg C]
# - Avg Wind Speed @ 6ft [m/s]
# - Albedo (CMP11)
# 
# Average to 1, 15 or 60 minutes and removes values out of bound (if any)
# 
# Save on PSM3 format for use with bifacialVF, SAM, and PVSyst

# In[1]:


datafolder = 'Data'


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import pvlib
import datetime
import pprint
import os


# In[3]:


plt.rcParams['timezone'] = 'Etc/GMT+7'
pd.plotting.register_matplotlib_converters()


# In[4]:


# This information helps with debugging and getting support :)
import sys, platform
print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)


# # Functions to Update SSRL Data and Save

# In[6]:


df = pd.read_csv(os.path.join(datafolder, 'BARNirrad.dat'), delimiter='\t')


# In[ ]:


for col in df:
    plt.figure()
    plt.plot(df[col])
    plt.title(col)


# In[10]:


loc_weatherdata_1T = weatherdata.tz_localize('Etc/GMT+7')

weatherdata_15T = _averageSRRL(loc_weatherdata_1T, interval='15T', closed='right', label='right')
weatherdata_60T = _averageSRRL(loc_weatherdata_1T, interval='60T', closed='right', label='right')



# In[11]:


freq='60T'
df = weatherdata_60T.copy()


# In[12]:


def fillYear(df, freq):
    import pandas as pd
    # add zeros for the rest of the year
    if freq is None:
        try:
            freq = pd.infer_freq(df.index)
        except:
            freq = '15T'  # 15 minute data by default
    tzinfo = df.index.tzinfo
    starttime = pd.to_datetime('%s-%s-%s %s:%s' % (df.index.year[0],1,1,0,0 ) ).tz_localize(tzinfo)
    endtime = pd.to_datetime('%s-%s-%s %s:%s' % (df.index.year[-1],12,31,23,60-int(freq[:-1])) ).tz_localize(tzinfo)
    beginning = df.index[0]
    ending = df.index[-1]
    df.loc[starttime] = 0  # set first datapt to zero to forward fill w zeros
    df.loc[endtime] = 0    # set last datapt to zero to forward fill w zeros
    df = df.sort_index()
    # add zeroes before data series
    df2= df[0:2].resample(freq).ffill()
    combined = pd.concat([df,df2],sort=True)
    combined = combined.loc[~combined.index.duplicated(keep='first')]
    # add zeroes after data series
    df2  = combined.resample(freq).bfill()
    return df2


# In[13]:


TMY = fillYear(weatherdata_60T, freq='60T')


# In[14]:


filterdates = (TMY.index >= '2023-1-1') & ~(is_leap_and_29Feb(TMY)) & (TMY.index < '2024-1-1') 
TMY = TMY[filterdates]
TMY


# In[15]:


saveSAM_SRRLWeatherFile(weatherdata_60T, os.path.join(weatherfolder,'PSM3_60T.csv'), includeminute=False) # No minutes = sunposition T-30min
saveSAM_SRRLWeatherFile(TMY, os.path.join(weatherfolder,'PSM3_TMY.csv'), includeminute=False) # No minutes = sunposition T-30min
saveSAM_SRRLWeatherFile(weatherdata, os.path.join(weatherfolder,'PSM3_1T.csv'), includeminute=False) # No minutes = sunposition T-30min
saveSAM_SRRLWeatherFile(weatherdata_15T, os.path.join(weatherfolder,'PSM3_15T.csv'), includeminute=False) # No minutes = sunposition T-30min

