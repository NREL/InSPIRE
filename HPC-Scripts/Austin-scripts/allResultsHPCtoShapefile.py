# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:14:12 2023

@author: akinzer
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

os.chdir('C:\\Users\\akinzer\\Documents\AUSTIN')

# In[1]

states = gpd.read_file('tl_2022_us_state/tl_2022_us_state.shp')

states = states.set_index('STUSPS').drop(index=['PR', 'VI', 'MP', 'GU', 'AS'])

conti = states.drop(index=['HI', 'AK'])
conti = conti.to_crs("EPSG:5070")
# colorado = states.loc[['CO']]
# california = states.loc[['CA']]
# california = california.to_crs(conti.crs)
# california.to_file("geo_california.shp")







# In[2]

os.chdir('C:\\Users\\akinzer\\Documents\AUSTIN\Runs_H5')
df = pd.read_pickle('Pitch_5_Colorado.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_co_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_co_h5.to_file("geo_co_h5.shp")


# In[3]

df = pd.read_pickle('Pitch_5_Arizona.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_az_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_az_h5.to_file("geo_az_h5.shp")


# In[4]

df = pd.read_pickle('Pitch_5_California.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_ca_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_ca_h5.to_file("geo_ca_h5.shp")


# In[5]

df = pd.read_pickle('Pitch_5_Idaho.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_id_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_id_h5.to_file("geo_id_h5.shp")


# In[6]

df = pd.read_pickle('Pitch_5_Montana.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_mt_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_mt_h5.to_file("geo_mt_h5.shp")


# In[7]

df = pd.read_pickle('Pitch_5_Nevada.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_nv_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_nv_h5.to_file("geo_nv_h5.shp")


# In[8]

df = pd.read_pickle('Pitch_5_New Mexico.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_nm_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_nm_h5.to_file("geo_nm_h5.shp")

# In[9]

df = pd.read_pickle('Pitch_5_Oregon.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_or_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_or_h5.to_file("geo_or_h5.shp")


# In[10]

df = pd.read_pickle('Pitch_5_Utah.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_ut_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_ut_h5.to_file("geo_ut_h5.shp")


# In[11]

df = pd.read_pickle('Pitch_5_Washington.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_wa_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_wa_h5.to_file("geo_wa_h5.shp")


# In[12]

df = pd.read_pickle('Pitch_5_Wyoming.pkl')


df = df.drop(columns=['MonthStart','MonthEnd',
                          'ResultPVWm2Back','ResultPVGround'])

df = df.rename(columns={'testbedAmean': 'A_mean', 
                        'testbedBmean': 'B_mean',
                        'testbedCmean': 'C_mean',
                        'testbedA_normGHI': 'A_normGHI', 
                        'testbedB_normGHI': 'B_normGHI', 
                        'testbedC_normGHI': 'C_normGHI'})

geo_wy_h5 = gpd.GeoDataFrame(df, 
                  geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                  crs = CRS('EPSG:4326')).to_crs(conti.crs)
geo_wy_h5.to_file("geo_wy_h5.shp")