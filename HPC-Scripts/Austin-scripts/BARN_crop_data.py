# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 14:39:19 2023

@author: akinzer
"""

# In[0]
# import packages
import os
import pandas as pd
import numpy as np
import plotly as plt
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot

os.chdir('C:\\Users\\akinzer\\Downloads')

# In[1]
# import excel file with raw crop data
xls = pd.ExcelFile('BARN_python_data3.xlsx')
df_fruit = pd.read_excel(xls, 'Tomato-Pepper-Master')
df_greens = pd.read_excel(xls, 'Leafy-Green-Master')
df_plant_count = pd.read_excel(xls, 'Plant Totals')


# In[2]
# clean up dataframes
# df_fruit = df_fruit.fillna(0)
df_fruit = df_fruit.rename(columns={'Unnamed: 0': 'Crop', 'tomatoes - Austin': 'Crop', 'Unnamed: 1': 'Variety', 'Unnamed: 2': 'Bed', 'Unnamed: 3': 'Plant #', 'Date': 'Observation'})
df_fruit = df_fruit.drop([1],axis=0)


df_plant_count = df_plant_count.drop(df_plant_count.columns[[3,4]],axis=1)
df_plant_count = df_plant_count.fillna(0)
df_fruit_norm = df_fruit.drop(df_fruit.columns[[5,6]],axis=1)

# df.loc[df['Courses'] == value]
# df.loc[(df['Discount'] >= 1000) & (df['Discount'] <= 2000)]

# In[4]
# normalize fruit/flower/buds

fruits = ['Tomato','Pepper']
varieties = [1,2]
beds = ['A','D','E','F']
observations = ['Buds','Flowers','Fruit','Marketable Fruit Count', 'Marketable Fruit Mass']
fruit_index = []
variety_index = []
bed_index = []
obs_index = []
bed_sum = []
bed_avg = []

for f in range(0,len(fruits)):
    fruit = fruits[f]
    for v in range(0,len(varieties)):
        variety = varieties[v]
        for b in range(0,len(beds)):
            bed = beds[b]
            for obs in range(0,len(observations)):
                observation = observations[obs]
                
                fruit_index.append(fruit)
                variety_index.append(variety)
                bed_index.append(bed)
                obs_index.append(observation)
                
                df1 = df_fruit_norm.loc[(df_fruit_norm['Crop'] == fruit) & 
                        (df_fruit_norm['Variety'] == variety) & 
                        (df_fruit_norm['Bed'] == bed) &
                        (df_fruit_norm['Observation'] == observation)]
    
                df_temp = df1.iloc[:,5:].sum()
                bed_sum.append(df_temp.values)
                
                df2 = df_plant_count.loc[(df_plant_count['Crop'] == fruit) & 
                        (df_plant_count['Variety'] == variety) & 
                        (df_plant_count['Bed'] == bed)]
                
                df_temp = df1.iloc[:,5:].sum() / df2.iloc[:,3:]
                bed_avg.append(df_temp.values)
                
df_all = pd.DataFrame(list(zip(fruit_index,variety_index, bed_index, obs_index, bed_sum, bed_avg)), 
                      columns=['fruit', 'variety', 'bed', 'observation','total','normalized'])



# need to clean up excel file, NaN values...
# add normalization for plant counts to harvest data, buds, etc.

# In[5]
# create dataframe for tomato1
days = df_fruit.iloc[0,7:] # set index for days since planting

tomato1 = pd.DataFrame(data={'bedAtotal' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
tomato1.set_index(days.values, inplace=True)

# tomato1bedD = pd.DataFrame(data=df_all.loc[(df_all['fruit'] == 'Tomato') &
#                                       (df_all['variety'] == 1) &
#                                       (df_all['bed'] == 'D') & 
#                                       (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'))
# tomato1bedD.set_index(days.values, inplace=True)

# plot normalized tomato fruit count
# tomato1.plot(x='Days since planting',
#         kind='bar',
#         stacked=False,
#         title='Grouped Bar Graph with dataframe')

# In[6]
# create dataframe for Tomato 2
tomato2 = pd.DataFrame(data={'bedAtotal' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
tomato2.set_index(days.values, inplace=True)

# In[7]
# create dataframe for Pepper 1

pepper1 = pd.DataFrame(data={'bedAtotal' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
pepper1.set_index(days.values, inplace=True)

# In[8]
# create dataframe for pepper 2

pepper2 = pd.DataFrame(data={'bedAtotal' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
pepper2.set_index(days.values, inplace=True)

# In[9]
# complete tomato 1 data

tomato1 = pd.DataFrame(data={'bedAtotal_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedDtotal_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedEtotal_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedFtotal_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedAnorm_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedDtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedEtotal_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedFtotal_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedAnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedDtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedEtotal_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedFtotal_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedAnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedDtotal_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedEtotal_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedFtotal_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedAnorm_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
tomato1.set_index(days.values, inplace=True)


# In[10]

tomato2 = pd.DataFrame(data={'bedAtotal_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedDtotal_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedEtotal_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedFtotal_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedAnorm_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_buds' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_buds': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedDtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedEtotal_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedFtotal_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedAnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_flowers': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedDtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedEtotal_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedFtotal_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedAnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_fruit': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_count' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_count': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedDtotal_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedEtotal_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedFtotal_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedAnorm_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_mass' : df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_mass': df_all.loc[(df_all['fruit'] == 'Tomato') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
tomato2.set_index(days.values, inplace=True)


# In[11]

pepper1 = pd.DataFrame(data={'bedAtotal_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedDtotal_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedEtotal_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedFtotal_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedAnorm_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedDtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedEtotal_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedFtotal_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedAnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedDtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedEtotal_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedFtotal_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedAnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedDtotal_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedEtotal_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedFtotal_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedAnorm_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 1) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
pepper1.set_index(days.values, inplace=True)


# In[12]

pepper2 = pd.DataFrame(data={'bedAtotal_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedDtotal_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedEtotal_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedFtotal_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['total'].explode('total'),
                             'bedAnorm_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_buds' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_buds': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Buds')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 1) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedDtotal_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedEtotal_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedFtotal_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['total'].explode('total'),
                             'bedAnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_flowers' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_flowers': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Flowers')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedDtotal_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedEtotal_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedFtotal_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['total'].explode('total'),
                             'bedAnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_fruit' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_fruit': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Fruit')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedDtotal_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedEtotal_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedFtotal_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['total'].explode('total'),
                             'bedAnorm_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_count' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_count': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Count')]['normalized'].explode('normalized').explode('normalized'),
                             
                             'bedAtotal_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedDtotal_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedEtotal_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedFtotal_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['total'].explode('total'),
                             'bedAnorm_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'A') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedDnorm_mass' : df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                 (df_all['variety'] == 2) &
                                                 (df_all['bed'] == 'D') & 
                                                 (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedEnorm_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'E') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                             'bedFnorm_mass': df_all.loc[(df_all['fruit'] == 'Pepper') &
                                                (df_all['variety'] == 2) &
                                                (df_all['bed'] == 'F') & 
                                                (df_all['observation'] == 'Marketable Fruit Mass')]['normalized'].explode('normalized').explode('normalized'),
                                 
                             })
pepper2.set_index(days.values, inplace=True)

