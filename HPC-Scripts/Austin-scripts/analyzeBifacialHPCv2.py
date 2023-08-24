# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:15:43 2023

last run on 3/16/2023
Colorado data with simple mapping and testbed irradiance analysis.

@author: akinzer
"""
# In[0]

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pyproj import CRS
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import ast 


os.chdir('C:\\Users\\akinzer\\Documents\\AUSTIN')
with open("locationlist", "r") as fp:
    b = json.load(fp)

hubheights = [1.5, 2.4] # meters
rtrs = [5.0, 10.0]# ,6.0] # meters
xgaps = [0.0, 1.0]# ,6.0] # meters
# locations = np.array(b) # choose specific locations for analysis
locations = [c for c in b if 'USA_CO' in c] # choose Colorado locations
locations = [l.replace('.', '__') for l in locations] # replace periods with double underscore for consistency
periods = np.array(['5TO5', '6TO6', '7TO7', '8TO8', '9TO9','5TO9']) # add more to match other time periods
tilts = np.array(['-50.0', '-45.0', '-40.0', '-35.0', '-30.0', '-25.0', '-20.0', '-15.0', '-10.0', '-5.0', '-0.0', 
                 '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '35.0', '40.0', '45.0', '50.0'])

# def clean(x):
#     return ast.literal_eval(x)

df1 = pd.read_csv('Results_CO.csv'
                  # # converters= {'xmod': clean,
                  #             'ymod': clean,
                  #             'zmod': clean,
                  #             'rearZ': clean,
                  #             'Wm2Front': clean,
                  #             'Wm2Back': clean,
                  #             'xground': clean,
                  #             'yground': clean,
                  #             'zground': clean,
                  #             'Wm2Ground': clean}
                      )

# In[1]

data_GHI = pd.read_csv('locationsEPW_info.csv')
df_temp = data_GHI["GHIs"].str.strip('[]').str.split(',', expand=True).astype(float)
data_GHI['5TO5'] = df_temp[0]
data_GHI['6TO6'] = df_temp[1]
data_GHI['7TO7'] = df_temp[2]
data_GHI['8TO8'] = df_temp[3]
data_GHI['9TO9'] = df_temp[4]
data_GHI['5TO9'] = df_temp[5]

data_GHI.set_index('location', inplace=True)
# GHI_val = data_GHI.loc['USA_CO_Aspen-Pitkin__County-Sardy__Field__724676_TMY3','5TO5']


# In[2]:
ground_vals = []
front_vals = []
back_vals = []
location_index = []
hubheight_index = []
rtr_index = []
xgap_index = []
period_index = []
bedA_vals = []
bedB_vals = []
bedC_vals = []
edge_vals = []
xground_index = []
GHI_vals = []
lat_vals = []
lon_vals = []
bedA_pct_vals = []
bedB_pct_vals = []
bedC_pct_vals = []
edge_pct_vals = []
fullrow_vals = []
fullrow_pct_vals = []
inside_vals = []
inside_pct_vals = []

for ii in range(0, len(locations)):
    location = locations[ii][:-5] # drop 'epw' on location name
    for jj in range(0, len(hubheights)):
        hubheight = hubheights[jj]
        for kk in range(0, len(rtrs)):
            rtr = rtrs[kk]
            for ll in range(0,len(xgaps)):
                xgap = xgaps[ll]
                for mm in range(0,len(periods)):
                    period = periods[mm]
                    
                    
                    df3 = df1.loc[(df1['location'] == location) & 
                            (df1['hubheight'] == hubheight) & 
                            (df1['rtr'] == rtr) & 
                            (df1['xgap'] == xgap) &
                            (df1['period'] == period)]

                    location_index.append(location)
                    hubheight_index.append(hubheight)
                    rtr_index.append(rtr)
                    xgap_index.append(xgap)
                    period_index.append(period)
                    
                    ghi_temp = data_GHI.loc[location,period]
                    df_temp = df3["Wm2Ground"].str.strip('[]').str.split(',', expand=True).astype(float)
                    df_temp = df_temp.sum(axis=0)
                    ground_vals.append(df_temp.values)
                    
                    fullrow = df_temp.mean()
                    fullrow_vals.append(fullrow)
                    fullrow_pct = fullrow / ghi_temp
                    fullrow_pct_vals.append(fullrow_pct)
                    
                    if hubheight == 1.5:
                        xp = 10
                    else:
                        xp = 20
                                        
                    dist1 = int(np.floor(len(df_temp[xp:-xp])/3))
                    testbedA = df_temp[xp:(xp + dist1)]
                    testbedA = testbedA.mean()
                    bedA_vals.append(testbedA)
                    bedA_pct = testbedA / ghi_temp
                    bedA_pct_vals.append(bedA_pct)
                    
                    testbedB = df_temp[(xp + dist1):(xp + dist1*2)]
                    testbedB = testbedB.mean()
                    bedB_vals.append(testbedB)
                    bedB_pct = testbedB / ghi_temp
                    bedB_pct_vals.append(bedB_pct)

                    testbedC = df_temp[(xp + dist1*2):-xp]
                    testbedC = testbedC.mean()
                    bedC_vals.append(testbedC)
                    bedC_pct = testbedC / ghi_temp
                    bedC_pct_vals.append(bedC_pct)
                    
                    edge = (df_temp[:xp].sum() + df_temp[-xp:].sum())/(xp*2)
                    edge_vals.append(edge)
                    edge_pct = edge / ghi_temp
                    edge_pct_vals.append(edge_pct)

                    inside = df_temp[xp:-xp].mean()
                    inside_vals.append(inside)
                    inside_pct = inside / ghi_temp
                    inside_pct_vals.append(inside_pct)
                    
                    df_temp = df3["Wm2Front"].str.strip('[]').str.split(',', expand=True).astype(float)
                    df_temp = df_temp.sum(axis=0)
                    df_temp = df_temp.mean()
                    front_vals.append(df_temp)
                    
                    df_temp = df3["Wm2Back"].str.strip('[]').str.split(',', expand=True).astype(float)
                    df_temp = df_temp.sum(axis=0)
                    df_temp = df_temp.mean()
                    back_vals.append(df_temp)
                    
                    GHI_vals.append(ghi_temp)
                    lat_vals.append(data_GHI.loc[location]['lat'])
                    lon_vals.append(data_GHI.loc[location]['lon'])
                    
                    
df_all = pd.DataFrame(list(zip(location_index,hubheight_index, rtr_index, xgap_index, period_index,
        front_vals,back_vals,ground_vals, bedA_vals, bedB_vals, bedC_vals, 
        edge_vals, GHI_vals, bedA_pct_vals, bedB_pct_vals, bedC_pct_vals, edge_pct_vals,
        fullrow_vals, fullrow_pct_vals, inside_vals, inside_pct_vals, lat_vals, lon_vals)), 
                      columns=['location', 'hubheight', 'rtr', 'xgap','period','Wm2Front','Wm2Back','Wm2Ground', 
                               'bedA_vals', 'bedB_vals', 'bedC_vals', 'edge_vals', 'GHI_val', 
                               'bedA_pct_vals', 'bedB_pct_vals', 'bedC_pct_vals', 'edge_pct_vals',
                               'fullrow_vals', 'fullrow_pct_vals', 'inside_vals', 'inside_pct_vals',
                               'lat', 'lon'])                

                    
# In[3]

data_GHI = pd.read_csv('locationsEPW_info.csv')
df_temp = data_GHI["GHIs"].str.strip('[]').str.split(',', expand=True).astype(float)
data_GHI['May'] = df_temp[0]
data_GHI['June'] = df_temp[1]
data_GHI['July'] = df_temp[2]
data_GHI['August'] = df_temp[3]
data_GHI['September'] = df_temp[4]
data_GHI['Season'] = df_temp[5]

df_temp = data_GHI.copy()
df_temp.set_index('location', inplace=True)
GHI_val = df_temp.loc['USA_CO_Aspen-Pitkin__County-Sardy__Field__724676_TMY3','May']

# In[4]

# begin mapping!

# In[5]:


# import the United States shape file
states = gpd.read_file('tl_2022_us_state/tl_2022_us_state.shp')
#states = states.to_crs("EPSG:5070")

# set state code as index, exclude states that we will never display
states = states.set_index('STUSPS').drop(index=['PR', 'VI', 'MP', 'GU', 'AS'])

conti = states.drop(index=['HI', 'AK'])
conti = conti.to_crs("EPSG:5070")

alaska = states.loc[['AK']]
hawaii = states.loc[['HI']]


# In[7]:


datafile = r'CO_data_prelim.csv'


# In[8]:


data = pd.read_csv(datafile)


# In[17]:


data.keys()


# In[23]:


df = data[(data['hubheight']==1.5) &
         (data['rtr'] == 5.0) & 
         (data['xgap'] == 0.0) &
          (data['period'] == '5TO5')]
df = df[['lat', 'lon', 'bedA_vals', 'bedB_vals', 'bedC_vals', 'edge_vals']].reset_index()


# In[24]:


df


# In[28]:


geo_conti = gpd.GeoDataFrame(df['bedA_vals'], 
                 geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])],
                 crs = CRS('EPSG:4326')).to_crs(conti.crs)


# In[29]:


geo_conti['bedA_vals'].quantile(q=0.98, interpolation='linear')


# In[31]:


fig, ax = plt.subplots(figsize=(4,4))
geo_conti['bedA_vals'].plot.hist(bins=1000, density=True, color='dimgray', label='norm. histogram')

ax.legend()

for rect in ax.patches:
    if rect.get_x() < 0.0:
        rect.set_color('firebrick')

#ax.set_xlim(-2.5, 16)
#ax.set_ylim(0, 0.225)
ax.set_xlabel('Ground Irradiance [W/m$^2$]')
ax.set_ylabel('Density')
ax.set_box_aspect(1)
#ax.set_ylim(0, 0.01)
#ax.set_xlim(100,600)
plt.tight_layout()
plt.savefig('asdf.png', dpi=600)
plt.savefig('asdf.pdf')


# In[34]:


# create an axis with 2 insets − this defines the inset sizes
fig, continental_ax = plt.subplots(figsize=(13, 8))


# Set bounds to fit desired areas in each plot
continental_ax.set_xlim(-2.5E6, 2.5E6)
#continental_ax.set_ylim(22, 53)
continental_ax.axis('off')

# Plot the data per area - requires passing the same choropleth parameters to each call
# because different data is used in each call, so automatically setting bounds won’t work

#vmin = 200.0
#vmax = 600.0

bound_plot = {'color':'gray', 'lw':0.75 }

conti.boundary.plot(ax=continental_ax, **bound_plot)

cont_plot = {'column':'bedA_vals', 'cmap':'viridis', 'marker': 'o', 'markersize': 0.1}
             # 'vmin':vmin, 'vmax':vmax, 'marker':'o', 'markersize':0.1} #
legend_kwds={'shrink':0.75, 'drawedges':False, 'label':'Ground Cumulative Irradiance [W/m$^2$]', #'ticks': np.linspace(0,15, 16), 
             'pad':0, 'aspect':30}

geo_conti.plot(ax=continental_ax, legend=True, legend_kwds=legend_kwds, **cont_plot)

#continental_ax.set_title('Ground Irradiance Testbed C in June', fontsize=20, y=0.95)

plt.tight_layout()
#plt.savefig('AgriPV_TestbedC_June.png', dpi=600)

# In[35]:
    
# export geodataframe to shapefile

# geo_conti.to_file("df_conti.shp")

