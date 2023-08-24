# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:37:34 2023

@author: akinzer
"""

# In [0]
import os
import numpy as np
import pandas as pd
import pickle

os.chdir('C:\\Users\\akinzer\\Documents\\AUSTIN')

# In[1]
df1 = pd.read_csv('Results_Clifton.csv')
df1 = df1.drop('yground',axis=1)
df1 = df1.drop('zground',axis=1)
df1 = df1.drop('index',axis=1)

# In[2]
df_temp = df1['Wm2Front'].str.strip('[]').str.split(',', expand=True).astype(float)

# In[3]
cols = df1['xground'][0].strip('[]').split(',')

# In[4]
# df_temp.set_axis(cols,axis='columns',inplace=True)

# In[5]
df_temp['timestamp'] = df1['timestamp']
df_temp['datetime'] = df1['datetime']
df_temp['month'] = df1['month']
df_temp['day'] = df1['day']
df_temp['time'] = df1['time']
df_temp['ghi'] = df1['ghi']

# In[6]
cols2 = np.arange(0,55)
rows2 = np.arange(0,2862)
df2 = pd.DataFrame(index = rows2, columns=cols2)
norm_ghi = []

# In[7]
for r in rows2:
    norm_ghi.append(df_temp.iloc[(r):(r+1),:55]/df_temp['ghi'][r])
    
# In[8]
df2 = pd.concat(norm_ghi)
df2.set_axis(cols,axis='columns',inplace=True)
df2['timestamp'] = df1['timestamp']
df2['datetime'] = df1['datetime']
df2['month'] = df1['month']
df2['day'] = df1['day']
df2['time'] = df1['time']
df2['ghi'] = df1['ghi']

# df2.to_csv('clifton_norm_ghi.csv')

