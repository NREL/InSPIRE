# -*- coding: utf-8 -*-
"""
@author: austi
"""

from pathlib import Path
import os
import bifacialvf
import matplotlib.pyplot as plt
# import pvlib
import pandas as pd
import plotly.express as px
from plotly.offline import plot

# IO Files
testfolder = Path().resolve().parent.parent / 'bifacialvf' / 'TEMP' / 'Tutorial_01'
if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    

plt.rcParams['timezone'] = 'Etc/GMT+7'
font = {'family' : 'DejaVu Sans',
'weight' : 'bold',
'size'   : 22}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (12, 5)

writefiletitle = os.path.join(testfolder, 'Results_Tutorial2.csv')


# Variables
lat = 39.748171540931274             # Golden CO, Coords.
lon = -105.22093948593452          # Golden CO, Coords.
tilt = 0                    # PV tilt (deg) - not used for tracking system
sazm = 180               # PV Azimuth(deg) or tracker axis direction
albedo = None               # Calculated in previous section from SRRL data. Value is 0.28 up to 11/18/19o
hub_height= 2/2            #1.5m / 2m collector width
pitch = 2/0.3/2              # 1 / 0.35 where 0.35 is gcr --- row to row spacing in normalized panel lengths. 
rowType = "interior"        # RowType(first interior last single)
transFactor = 0             # TransmissionFactor(open area fraction)
sensorsy = 12                # sensorsy(# hor rows in panel)   <--> THIS ASSUMES LANDSCAPE ORIENTATION 
PVfrontSurface = "glass"    # PVfrontSurface(glass or ARglass)
PVbackSurface = "glass"     # PVbackSurface(glass or ARglass)
agriPV = True


# Tracking instructions
tracking=True
backtrack=True
limit_angle = 45

# Download and Read input
TMYtoread=bifacialvf.getEPW(lat=lat,lon=lon, path = testfolder)
myTMY3, meta = bifacialvf.readInputTMY(TMYtoread)
deltastyle = 'TMY3'
#myTMY3 = myTMY3.iloc[0:24].copy()  # Simulate just the first 24 hours of the data file for speed on this example

bifacialvf.simulate(myTMY3, meta, writefiletitle=writefiletitle, 
         tilt=tilt, sazm=sazm, pitch=pitch, hub_height=hub_height, 
         rowType=rowType, transFactor=transFactor, sensorsy=sensorsy, 
         PVfrontSurface=PVfrontSurface, PVbackSurface=PVbackSurface, 
         albedo=albedo, tracking=tracking, backtrack=backtrack, 
         limit_angle=limit_angle, deltastyle=deltastyle, agriPV=agriPV)

#Load the results from the resultfile
from bifacialvf import loadVFresults
(data, metadata) = loadVFresults(writefiletitle)

data.set_index(pd.to_datetime(data['date']), inplace=True, drop=True)
data.index = data.index.map(lambda t: t.replace(year=2021))   # Chagning to be the same year
groundIrrads = data['Ground Irradiance Values'].str.strip('[]').str.split(' ', expand=True).astype(float)

df = groundIrrads.groupby([pd.Grouper(freq='M')]).sum().T
# I'm sure there's a fancier way to do this but hey, this works.
df.rename(columns={ df.columns[0]: "Jan", df.columns[1]: "Feb",df.columns[2]: "Mar", df.columns[3]: "Apr",
                   df.columns[4]: "May",df.columns[5]: "June",df.columns[6]: "July",df.columns[7]: "Aug",
                   df.columns[8]: "Sept",df.columns[9]: "Oct",df.columns[10]: "Nov",df.columns[11]: "Dec"    }, inplace = True)

fig = (df/1000).plot()
fig.set_xlabel('Position between start of row and next row [%]')
fig.set_ylabel('Cumulative Insolation for the month [kWh/m2]')

monthlyGHI = pd.DataFrame(data['ghi'].groupby([pd.Grouper(freq='M')]).sum())

monthlyGHI['Month'] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
monthlyGHI.set_index(monthlyGHI['Month'], inplace=True)
monthlyGHI = monthlyGHI.drop(columns='Month')

growingSeasonGHI = pd.DataFrame(monthlyGHI[3:9].sum())
gs_GroundIrrads = pd.DataFrame(groundIrrads['2021-04-01':'2021-10-01'].sum()).T
gs_GroundIrrads_norm = pd.DataFrame(gs_GroundIrrads.divide(growingSeasonGHI.iloc[0], axis=0))

df_norm = df.T
df_norm = df_norm.div(monthlyGHI['ghi'], axis=0)
df_norm = df_norm.T

fig = (df_norm).plot()
fig.set_xlabel('Position between start of row and next row [%]')
fig.set_ylabel('Normalized Insolation for the month/GHI')


fig = px.imshow(df.transpose(),
                labels=dict(x="Position from rear edge of row to rear edge of next row (%)", y="Month", color="Cumulative Insolation (kWh/m2)"),
                # x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                )
fig.update_xaxes(side="top")
plot(fig)

fig1 = px.imshow(df_norm.transpose(),
                labels=dict(x="Position from rear edge of row to rear edge of next row (%)", y="Month", color="Normalized Insolation"),
                )
# plot(fig1)

# plot single day hourly heatmap

# fig2 = px.imshow(groundIrrads['2021-01-01'])              # choose date
# fig2.update_layout(xaxis = dict(autorange="reversed"))    # reverse x-axis
# plot(fig2)                                                # output to browser


# calculate average front and back global tilted irradiance across the module chord
data['GTIFrontavg'] = data[['No_1_RowFrontGTI', 'No_2_RowFrontGTI','No_3_RowFrontGTI','No_4_RowFrontGTI','No_5_RowFrontGTI','No_6_RowFrontGTI']].mean(axis=1)
data['GTIBackavg'] = data[['No_1_RowBackGTI', 'No_2_RowBackGTI','No_3_RowBackGTI','No_4_RowBackGTI','No_5_RowBackGTI','No_6_RowBackGTI']].mean(axis=1)


# Print the annual bifacial ratio
frontIrrSum = data['GTIFrontavg'].sum()
backIrrSum = data['GTIBackavg'].sum()
# print('The bifacial ratio for ground clearance {} and rtr spacing {} is: {:.1f}%'.format(clearance_height,pitch,backIrrSum/frontIrrSum*100))



data


# fig5 = px.imshow(finalData,
#                 labels=dict(x="Percent of row-to-row distance", y="Normalized Clearance Height", color="Normalized Insolation"),
#                 y = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
#                 aspect="auto", origin="lower"
#                 )


# Plot some results 



# plot the rear irradiance distribution for a single point in time.

get_ipython().run_line_magic('matplotlib', 'inline')

data['GTIBackstd'] = data[['No_1_RowBackGTI', 'No_2_RowBackGTI','No_3_RowBackGTI','No_4_RowBackGTI','No_5_RowBackGTI','No_6_RowBackGTI']].std(axis=1)
data.set_index(pd.to_datetime(data['date']), inplace=True, drop=True)
data.index = data.index.map(lambda t: t.replace(year=2021))   # Chagning to be the same year
singleday = (data.index > '2021-07-09') & (data.index<'2021-07-10')
singleday2 = (data.index > '2021-07-15') & (data.index<'2021-07-16')

fig3, ax = plt.subplots()
ax1 = ax.twinx()
ax1.plot(data.index[singleday],data['GTIFrontavg'][singleday],'k')
ax1.set_ylabel('Front Irradiance (Wm-2)')
ax.set_ylabel('Rear Irradiance (Wm-2)')
ax.plot(data.index[singleday], data['No_1_RowBackGTI'][singleday],'r' , alpha =0.5)
ax.plot(data.index[singleday], data['No_2_RowBackGTI'][singleday], 'b', alpha = 0.5)
ax.plot(data.index[singleday], data['No_6_RowBackGTI'][singleday], 'g', alpha = 0.5)
ax.set_title('Sunny day')
fig3.autofmt_xdate()
fig3.tight_layout()
fig3


# fig4, ax2 = plt.subplots()
# ax3 = ax2.twinx()
# ax3.plot(data.index[singleday2],data['GTIFrontavg'][singleday2],'k')
# ax3.set_ylabel('Front Irradiance (Wm-2)')
# ax2.set_ylabel('Rear Irradiance (Wm-2)')
# ax2.plot(data.index[singleday2], data['No_1_RowBackGTI'][singleday2],'r' , alpha =0.5)
# ax2.plot(data.index[singleday2], data['No_2_RowBackGTI'][singleday2], 'b', alpha = 0.5)
# ax2.plot(data.index[singleday2], data['No_6_RowBackGTI'][singleday2], 'g', alpha = 0.5)
# ax2.set_title('Cloudy day')
# fig4.autofmt_xdate()
# fig4.tight_layout()
# fig4


