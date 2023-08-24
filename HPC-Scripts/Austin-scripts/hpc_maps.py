# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:49:29 2023

@author: akinzer
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import numpy as np

USmap = True
if USmap:
    states = gpd.read_file('../tl_2022_us_state/tl_2022_us_state.shp')
else: # Testing options for PR maps specifically
    #states = gpd.read_file('PR_coastline/PR_coastline.shp') # doestn seem to show up the line?
    #states = gpd.read_file('tl_2022_72_place/tl_2022_72_place.shp')  # lots of little... counties? idk, weird. right lat/lon
    #states = gpd.read_file('stanford-xv279yj9196-shapefile/xv279yj9196.shp') # no line either . right lat/lon
    states = states.to_crs("EPSG:5070")
#datafile = r'ALLSetups_new_Puerto Rico.csv'
datafile = r'C:\Users\sayala\Box\AGRIPVWORLDPICKLES\ALLSetups_June7_Puerto Rico.csv'
data = pd.read_csv(datafile)
data['setup'].unique()

daily = []

for setup in range(1, 6):
    df = data[(data['setup']==setup)]
    a = df.groupby('gid')['ghi_sum', 'insidemean'].sum()
    daily.append(list(a['insidemean']/a['ghi_sum']))
    gidlist = a.index


type(daily)
df = pd.DataFrame(daily)
df = df.T
df.columns = ['Setup1', 'Setup2', 'Setup3', 'Setup4', 'Setup5']
df.index = gidlist


latslons = data.groupby('gid')['latitude','longitude'].mean()


df['latitude'] = latslons['latitude']
df['longitude'] = latslons['longitude']
df.keys()

testbeds = ['Setup1', 'Setup2', 'Setup3', 'Setup4', 'Setup5']
vmin = np.round(df[testbeds].min().min(),2)
vmax = np.round(df[testbeds].max().max(),2)
# create an axis with 2 insets − this defines the inset sizes

for testbed in testbeds:

    geo_conti = gpd.GeoDataFrame(df[testbed], 
                     geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])],
                     crs = CRS('EPSG:4326')).to_crs(states.crs)


    fig, continental_ax = plt.subplots(figsize=(8, 8))


    # Set bounds to fit desired areas in each plot
    #continental_ax.set_xlim(3E6, 3.4E6)
    continental_ax.set_xlim(-67.5, -65)
    continental_ax.set_ylim(17.5, 19)
    continental_ax.axis('off')

    # Plot the data per area - requires passing the same choropleth parameters to each call
    # because different data is used in each call, so automatically setting bounds won’t work

    #vmin = 200.0
    #vmax = 600.0

    bound_plot = {'color':'gray', 'lw':0.75 }

    #states.boundary.plot(ax=continental_ax, **bound_plot)


    cont_plot = {'column':testbed, 'cmap':'viridis', 'marker': 's', 'markersize': 52, 'facecolor': 'b',
                 'vmin':vmin, 'vmax':vmax} #
    legend_kwds={'shrink':0.75, 'drawedges':False, 'label':'Ground Cumulative Irradiance [W/m$^2$]', #'ticks': np.linspace(0,15, 16), 
                 'pad':0, 'aspect':30}

    geo_conti.plot(ax=continental_ax, legend=True, legend_kwds=legend_kwds, **cont_plot)

    #continental_ax.set_title('Ground Irradiance Testbed C in June', fontsize=20, y=0.95)

    plt.title(testbed)
    plt.tight_layout()
    #plt.savefig('AgriPV_TestbedC_June.png', dpi=600)

