# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:34:36 2023

@author: akinzer

HPC Spatial Irradiance Site Comparisons

"""

# In[0]
# import packages and set directory

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import ast 
import pickle

os.chdir('C:\\Users\\akinzer\\Documents\AUSTIN\AllSetups_April10')

# In[1]
df_tx = pd.read_pickle('ALLSetups_Texas.pkl')
df_ca = pd.read_pickle('ALLSetups_California.pkl')
df_co = pd.read_pickle('ALLSetups_Colorado.pkl')
df_or = pd.read_pickle('ALLSetups_Oregon.pkl')
df_az = pd.read_pickle('ALLSetups_Arizona.pkl')


# In[2]

lat = 35.29
lon = -118.94
bakersfield = df_ca[(df_ca['latitude'].round(2) == lat) & (df_ca['longitude'].round(2) == lon)]

lat = 39.09
lon = -108.58
grandjunction = df_co[(df_co['latitude'].round(2) == lat) & (df_co['longitude'].round(2) == lon)]

lat = 32.13
lon = -110.94
tuscon = df_az[(df_az['latitude'].round(2) == lat) & (df_az['longitude'].round(2) == lon)]

lat = 44.89
lon = -122.94
salem = df_or[(df_or['latitude'].round(2) == lat) & (df_or['longitude'].round(2) == lon)]

lat = 29.53
lon = -98.62
sanantonio = df_tx[(df_tx['latitude'].round(2) == lat) & (df_tx['longitude'].round(2) == lon)]


# In[3]

bakersfield.to_csv("bakersfield.csv")
grandjunction.to_csv("grandjunction.csv")
tuscon.to_csv("tuscon.csv")
salem.to_csv("salem.csv")
sanantonio.to_csv("sanantonio.csv")


# In[4]

os.chdir('C:\\Users\\akinzer\\Documents\AUSTIN')

states = gpd.read_file('tl_2022_us_state/tl_2022_us_state.shp')

states = states.set_index('STUSPS').drop(index=['PR', 'VI', 'MP', 'GU', 'AS'])

conti = states.drop(index=['HI', 'AK'])
conti = conti.to_crs("EPSG:5070")

# In[5]

os.chdir('C:\\Users\\akinzer\\Documents\AUSTIN\AllSetups_April10')
locations = [bakersfield, grandjunction, tuscon, salem, sanantonio]
df_all = pd.concat(locations)


# In[6]

df_all = df_all.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df_all = df_all.rename(columns={'testbedA1mean': 'A1_mean', 'testbedA2mean': 'A2_mean', 'testbedA3mean': 'A3_mean', 
                        'testbedB1mean': 'B1_mean', 'testbedB2mean': 'B2_mean', 'testbedB3mean': 'B3_mean',
                        'testbedC1mean': 'C1_mean', 'testbedC2mean': 'C2_mean', 'testbedC3mean': 'C3_mean',
                        'testbedA1_normGHI': 'A1_normGHI', 'testbedA2_normGHI': 'A2_normGHI', 'testbedA3_normGHI': 'A3_normGHI',
                        'testbedB1_normGHI': 'B1_normGHI', 'testbedB2_normGHI': 'B2_normGHI', 'testbedB3_normGHI': 'B3_normGHI',
                        'testbedC1_normGHI': 'C1_normGHI', 'testbedC2_normGHI': 'C2_normGHI', 'testbedC3_normGHI': 'C3_normGHI'})

geo_comparison = gpd.GeoDataFrame(df_all, 
                                  geometry = [Point(xy) for xy in zip(df_all['longitude'], df_all['latitude'])],
                                  crs = CRS('EPSG:4326')).to_crs(conti.crs)

geo_comparison.to_file("geo_comparison.shp")