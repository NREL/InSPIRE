#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
from pathlib import Path

testfolder = 'TEMP' 
if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)


# In[13]:


from bifacial_radiance import *
import numpy as np
import datetime
import pandas as pd
import calendar


# In[7]:


# This information helps with debugging and getting support :)
import sys, platform
import bifacial_radiance as br
print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("bifacial_radiance version ", br.__version__)


# In[8]:


import seaborn as sns


# In[14]:


startdts = [datetime.datetime(2001,4,1,0),
            datetime.datetime(2001,5,1,0),
            datetime.datetime(2001,6,1,0),
            datetime.datetime(2001,7,1,0),
            datetime.datetime(2001,8,1,0),
            datetime.datetime(2001,9,1,0),
            datetime.datetime(2001,4,1,0),
            datetime.datetime(2001,10,1,0),]

enddts = [datetime.datetime(2001,5,1,0),
          datetime.datetime(2001,6,1,0),
          datetime.datetime(2001,7,1,0),
          datetime.datetime(2001,8,1,0),
          datetime.datetime(2001,9,1,0),
          datetime.datetime(2001,10,1,0),
          datetime.datetime(2001,10,1,0),
          datetime.datetime(2001,10,15,0),]

lat = 44.57615187732146   
lon = -123.23914850912513
clearance_heights = [0.88, 0.9482582, 0.6985] # m
ygaps = [0.02, 0.02, 0.02] # m
cws = [3.3655, 3.3655, 3.9624] # m
rtrs = [6.223, 8.4201, 6.8453] # m
tilt = 25
sazm = 180
albedo = 0.2 # 'grass'

# Field size. Just going for 'steady state'
nMods = 20
nRows = 7


# In[ ]:





# In[17]:


hub_heights = [4.3, 3.5, 2.5, 1.5]
results_BGG=[]
results_GFront=[]
results_GRear=[]
results_GGround=[]
results_coordY=[]
setups = []
months = []
results_GHI = []
for ii in range(0, len(clearance_heights)):
    for jj in range(0, len(startdts)):
        
        if jj == (len(startdts)-2):
            months.append('Season')
        elif jj == (len(startdts)-1): 
            months.append('October1-15')
        else:
            months.append(calendar.month_abbr[jj+4])
        setups.append(ii+1)
        # irr_GROUND_Month_6_setup_1_Row4_Module10_Back.csv
        fileground= os.path.join(testfolder, 'results', f'irr_GROUND_Month_'+str(jj+4)+'_setup_'+str(ii+1)+'_Row4_Module10_Front.csv')
        filepv= os.path.join(testfolder, 'results', f'irr_MODULE_Month_'+str(jj+4)+'_setup_'+str(ii+1)+'_Row4_Module10.csv')
        resultsGround = load.read1Result(fileground)
        resultsPV = load.read1Result(filepv)
        #  resultsDF = load.cleanResult(resultsDF).dropna() # I checked them they are good because even number of sensors
        results_GGround.append(list(resultsGround['Wm2Front']))
        results_coordY.append(list(resultsGround['y']))
        results_GFront.append(list(resultsPV['Wm2Front']))
        results_GRear.append(list(resultsPV['Wm2Back']))
        results_BGG.append(resultsPV['Wm2Back'].sum()*100/resultsPV['Wm2Front'].sum())


# In[21]:


df = pd.DataFrame(list(zip(setups, months, results_coordY, results_GGround,
                          results_GFront, results_GRear, results_BGG)),
               columns =['Setup', 'Month', 'GroundCoordY', 'Gground', 'Gfront', 'Grear', 'BGG'])


# In[23]:


# Example of selectiong one setup one month
foo = df[(df['Setup']==1) & (df['Month']=='Apr')]


# In[106]:


import matplotlib.pyplot as plt


# In[129]:


#Finding min and max
minground = 100000
maxground = 0
setups=[1,2,3]
mmonths=list(df['Month'].unique())

for ii in range(0, len(setups)):
    for jj in range(0, len(mmonths)-2):
        setup = setups[ii]
        month = mmonths[jj]
        foo = df[(df['Setup']==setup) & (df['Month']==month)]
        
        if minground > min(foo['Gground'].iloc[0]):
            minground = min(foo['Gground'].iloc[0])
        
        if maxground < max(foo['Gground'].iloc[0]):
            maxground = max(foo['Gground'].iloc[0])
                
print(minground, maxground)
minground = 0
maxground = 220000


# In[132]:


setups=[1,2,3]
mmonths=list(df['Month'].unique())

for ii in range(0, len(setups)):
    for jj in range(0, len(mmonths)-2):
        plt.figure()
        setup = setups[ii]
        month = mmonths[jj]
        foo = df[(df['Setup']==setup) & (df['Month']==month)]
            
        foo2=pd.DataFrame(foo['Gground'].iloc[0])
        foo3=foo2.T
        foo3

        foo2=pd.DataFrame(foo['GroundCoordY'].iloc[0])
        foo4=foo2.T
        foo4 = (foo4*100).round()
        foo3.columns=foo4.iloc[0].values

        ax=sns.heatmap(foo3, cmap='hot', vmin=minground, vmax=maxground, annot=False)
        ax.set_yticks([])
        #ax.set_xticks([])
        ax.set_ylabel('')  
        #ax.set_xlabel('')
        
        mytitle = 'Setup_'+str(setup)+', Mont_ '+str(month)
        ax.set_title(mytitle)
        plt.savefig(mytitle)


# In[ ]:




