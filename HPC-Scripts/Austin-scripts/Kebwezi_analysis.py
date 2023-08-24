# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:54:04 2023

@author: akinzer
"""

import os
import pandas as pd
import numpy as np
import plotly as plt
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

os.chdir('C:\\Users\\akinzer\\Documents\\Technical_Assistance\\Kebwezi')

# In[1]
# import csv files with raw data
df1 = pd.read_csv('Results_Kebwezi1.csv')
data_GHI = pd.read_csv('KenyaGHI.csv')

df1 = df1.drop(columns=['Unnamed: 0'])
data_GHI = data_GHI.drop(columns=['Time index'])
annual_GHI = data_GHI['GHI'].sum()

# In[2]:

# position_index = []
# rearZ_vals = []
# xground_index = []
# yground_index = []
# zground_index = []
rows = np.array(range(0,100))
cols = np.array(range(0,100))
GHI_vals = pd.DataFrame(index = rows, columns = cols)
GHI_norm_vals = pd.DataFrame(index = rows, columns = cols)

positions = np.array(range(0,100))


for i in positions:
    position = positions[i]                  
    
    df2 = df1.loc[(df1['position'] == position)]
    df_temp = df2['Wm2Front'].str.strip('[]').str.split(',', expand=True).astype(float)
    GHI_vals.iloc[i] = df_temp
    
    df_temp = df_temp / annual_GHI
    GHI_norm_vals.iloc[i] = df_temp
    


# df_temp = .str.strip('[]').str.split(',', expand=True).astype(float)

# In[3]:

#GHI_norm_vals.to_csv('GHI_norm_vals_Kebwezi.csv')
#GHI_vals.to_csv('GHI_vals.csv')    

# update dataframe index for heat map to display properly
#GHI_norm_vals = pd.read_csv('GHI_norm_vals_Kebwezi.csv',index_col=0)


# In[4]:

fig = px.imshow(GHI_norm_vals, origin="lower", aspect='equal')
fig.update_coloraxes(cmax=1, cmin=0, colorbar_dtick=0.05)
fig.layout.height = 750 
fig.layout.width = 800


fig.show()
plot(fig)


# In[5]:

fig = px.imshow(GHI_vals, origin="lower", aspect='equal')
#fig.update_coloraxes(cmax=0.65, cmin=0.45, colorbar_dtick=0.05)
fig.layout.height = 750 
fig.layout.width = 800


fig.show()
plot(fig)



# In[6]:



#GHI_norm_vals.to_csv('GHI_norm_vals_Kebwezi.csv')
#GHI_vals.to_csv('GHI_vals.csv')
    

