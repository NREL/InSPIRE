# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:53:36 2023

@author: akinzer
"""

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

os.chdir('C:\\Users\\akinzer\\Documents\AUSTIN')

# In[1]

states = gpd.read_file('tl_2022_us_state/tl_2022_us_state.shp')

# states = states.set_index('STUSPS').drop(index=['PR', 'VI', 'MP', 'GU', 'AS'])
states = states.set_index('STUSPS').drop(index=['PR'])

conti = states.drop(index=['HI', 'AK'])
conti = conti.to_crs("EPSG:5070")


# In[2]
os.chdir('C:\\Users\\akinzer\\Documents\AUSTIN')
df = pd.read_csv('ALLSetups_new_Puerto Rico.csv')

# In[3]

df = df.drop(columns=['DATE','Date_Localized',
                          'ResultPVGround'])

df = df.rename(columns={'testbedA1mean': 'A1_mean', 
                        'testbedB1mean': 'B1_mean',
                        'testbedC1mean': 'C1_mean',
                        'testbedA1_normGHI': 'A1_normGHI', 
                        'testbedB1_normGHI': 'B1_normGHI', 
                        'testbedC1_normGHI': 'C1_normGHI',
                        'testbedA2mean': 'A2_mean', 
                        'testbedB2mean': 'B2_mean',
                        'testbedC2mean': 'C2_mean',
                        'testbedA2_normGHI': 'A2_normGHI', 
                        'testbedB2_normGHI': 'B2_normGHI', 
                        'testbedC2_normGHI': 'C2_normGHI',
                        'testbedA3mean': 'A3_mean', 
                        'testbedB3mean': 'B3_mean',
                        'testbedC3mean': 'C3_mean',
                        'testbedA3_normGHI': 'A3_normGHI', 
                        'testbedB3_normGHI': 'B3_normGHI', 
                        'testbedC3_normGHI': 'C3_normGHI'})

setup1 = df[(df['setup'] == 1)]
setup2 = df[(df['setup'] == 2)]
setup3 = df[(df['setup'] == 3)]
setup4 = df[(df['setup'] == 4)]
setup5 = df[(df['setup'] == 5)]
setup6 = df[(df['setup'] == 6)]
setup7 = df[(df['setup'] == 7)]
setup8 = df[(df['setup'] == 8)]
setup9 = df[(df['setup'] == 9)]
setup10 = df[(df['setup'] == 10)]

pr1 = gpd.GeoDataFrame(setup1, 
                  geometry = [Point(xy) for xy in zip(setup1['longitude'], setup1['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr1.to_file("pr1.shp")

pr2 = gpd.GeoDataFrame(setup2, 
                  geometry = [Point(xy) for xy in zip(setup2['longitude'], setup2['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr2.to_file("pr2.shp")

pr3 = gpd.GeoDataFrame(setup3, 
                  geometry = [Point(xy) for xy in zip(setup3['longitude'], setup3['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr3.to_file("pr3.shp")

pr4 = gpd.GeoDataFrame(setup4, 
                  geometry = [Point(xy) for xy in zip(setup4['longitude'], setup4['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr4.to_file("pr4.shp")

pr5 = gpd.GeoDataFrame(setup5, 
                  geometry = [Point(xy) for xy in zip(setup5['longitude'], setup5['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr5.to_file("pr5.shp")

pr6 = gpd.GeoDataFrame(setup6, 
                  geometry = [Point(xy) for xy in zip(setup6['longitude'], setup6['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr6.to_file("pr6.shp")

pr7 = gpd.GeoDataFrame(setup7, 
                  geometry = [Point(xy) for xy in zip(setup7['longitude'], setup7['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr7.to_file("pr7.shp")

pr8 = gpd.GeoDataFrame(setup8, 
                  geometry = [Point(xy) for xy in zip(setup8['longitude'], setup8['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr8.to_file("pr8.shp")

pr9 = gpd.GeoDataFrame(setup9, 
                  geometry = [Point(xy) for xy in zip(setup9['longitude'], setup9['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr9.to_file("pr9.shp")

pr10 = gpd.GeoDataFrame(setup10, 
                  geometry = [Point(xy) for xy in zip(setup10['longitude'], setup10['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
pr10.to_file("pr10.shp")

