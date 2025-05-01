# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# This array is defined from the east rather than the west side, because of alignment with the torque tube on the east side, for the south row. Spacing is consistent with the south rather than the north row.

# +
import os
from pathlib import Path

testfolder = str(Path().resolve().parent / 'TEMP' /  'MA')

datafolder = str(Path().resolve().parent / 'NSRDB data')

if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)

results_dir = str(Path().resolve().parent / 'Results_graphs' / 'Irradiance modeling')

if not os.path.exists(results_dir):
    os.makedirs(results_dir)
# -

import bifacial_radiance as br
import numpy as np
import pandas as pd
import pvlib
import sys, platform
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import seaborn as sns
import time
import calendar


# This information helps with debugging and getting support :)
print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("bifacial_radiance version ", br.__version__)
print("pvlib version ", pvlib.__version__)

# +
simulationname = 'UMass'

# Location: UMass Crop Animal Research and Education Center
lat = 42.4783 
lon = -72.5779  

ft2m = 0.3048
in2m = 0.0254

# MakeModule Parameters
# Sharp NU-U235F1
# https://www.solarelectricsupply.com/sharp-nu-u235f1-solar-panels
moduletype='pv-collector'
numpanels = 3  # AgriPV site has 3 modules along the y direction
x = 64.60*in2m# m, From spec sheet
y = 39.10*in2m # m, From spec sheet
# if x > y, landscape. if x < y, portrait

# Scene Parameters:
azimuth_ang=180 # Facing south
tilt = 40 # tilt.
pitch = 36*ft2m # m From measurement
albedo = 0.2  #'grass'     # ground albedo
nRows = 2
n_piles = 11

# Scene Object 1: South row, 5 ft. section on the East (4 clusters)
nMods_1 = 4
xgap_1 = 5*ft2m # 5 ft converted to meters now
xcoord_1 = 0
ycoord_1 = 0

x_spacer = 3.4*ft2m # Transitional gap between 1st and 2nd sections

# Scene Object 2: South row, 3 ft. section (next 3 clusters)
nMods_2 = 3
xgap_2 = 3*ft2m # ft
ycoord_2 = 0 
xcoord_2 = -1*(x*(nMods_1-1)/2+ xgap_1 + x_spacer + xgap_2 + x*nMods_2/2)

x_spacer_2 = 4.35*ft2m # Transitional gap between 2nd and 3rd sections

# Scene Object 3: South row, 4 ft. section (3 clusters)
nMods_3 = 3
xgap_3 = 4*ft2m # ft
ycoord_3 = 0
xcoord_3 = xcoord_2 - (x*nMods_2/2 + xgap_2 + x_spacer_2 + xgap_3 + x*nMods_3/2) # xcoord_2 already negative

# Scene Object 4: South row, 2 ft. section (2 clusters)
nMods_4 = 2
xgap_4 = 2*ft2m # ft
ycoord_4 = 0
xcoord_4 = xcoord_3 - (x*nMods_3/2 + xgap_3 + xgap_4*2 + x*(nMods_4 - 1/2)) # xcoord_3 already negative

# TorqueTube Parameters
tubetype='square' # Other options: 'square' , 'hex'
material = 'Metal_Grey' # Other options: 'black'
diameter = 0.1 # m
zgap = 0.05 # m, pretty close to panel thickness of 1.80 in
torquetube_xcoord = xcoord_1 + 2.5*x + 2*xgap_1 # starting coordinate At the Eastern edge
torquetubelength = 115*ft2m # length to the western edge: ~105 ft

# set up sensor grid
# y positive is north
# x positive is East
yinc = 0.1 # 10 cm
sx_xinc = -0.1 # 10 cm, This moves xstart once every loop
# starting locations
# 7.2 m, east side
xstart = torquetube_xcoord + abs(sx_xinc)/2 # Start 5 cm past eastern edge
# -1.9 m, 
ystart = -y*numpanels/2 - 0.45 # Start a bit South of the first row
# Number
sensorsx = math.ceil((torquetubelength + abs(sx_xinc))/abs(sx_xinc)) # at least every 10 cm
sensorsy = math.ceil((y*numpanels*2 + pitch + 1)/yinc) # bed length + pitch + buffer

# Pile parameters
# based on photograph, looks approximately centered under right hand most panel
pile1x = torquetube_xcoord - x/2
pilesep = -11*ft2m 

# Calculate the hub height from available measurements
height_to_lower_edge = 7.5*ft2m
height_to_cluster_center = height_to_lower_edge + numpanels*y*math.sin(math.radians(tilt))/2
hub_height = height_to_cluster_center - zgap - diameter/2

#Add torquetube so parameters are accounted for During module definition, but
# set visibility to false Because single tube Will be custom Defined later
tubeParams = {'tubetype':tubetype,
              'diameter':diameter,
              'material':material,
              'axisofrotation':False,
              'visible':False}

# -


# bed extents (Minimum and maximum X values for Each test bed)
# bed minimum and maximum Y values are the same, all Defined for the south row
midx = pd.MultiIndex.from_product([[2, 3, 4, 5], ["West", "Center", "East"]])
col = ['xmin', 'xmax']
# fourth section is 2 foot, Second section is 3 foot, third section is 4 foot, first section is 5 foot
limit_data = np.array([[xcoord_4 + x/2, xcoord_4 + x/2 + xgap_4],
                       [xcoord_4 + x/2 + xgap_4, xcoord_4 + x/2 + xgap_4 + x],
                       [xcoord_4 + x/2 + xgap_4 + x, xcoord_4 + x/2 + 2*xgap_4 + x ],
                       [xcoord_2 - x/2 - xgap_2, xcoord_2 - x/2],
                       [xcoord_2 - x/2, xcoord_2 + x/2],
                       [xcoord_2 + x/2, xcoord_2 + x/2 + xgap_2],
                       [xcoord_3 - x/2 - xgap_3, xcoord_3 - x/2],
                       [xcoord_3 - x/2, xcoord_3 + x/2],
                       [xcoord_3 + x/2, xcoord_3 + x/2 + xgap_3],
                       [xcoord_1 - x/2 - xgap_1, xcoord_1 - x/2],
                       [xcoord_1 - x/2, xcoord_1 + x/2],
                       [xcoord_1 + x/2, xcoord_1 + x/2 + xgap_1]
                      ])
bed_x_limits = pd.DataFrame(limit_data, midx, col)

# project the panels down onto the ground to get the bed length
y_proj = numpanels * y * math.cos(math.radians(tilt))
bed_y_limits = {'ymin': 0, # Start at the post, to also exclude the pile from calculations
               'ymax': y_proj/2}

year = "2017"


def make_radobj_by_timeperiod(year, month = "season"):
    metadata = pd.read_csv(os.path.join(datafolder, "1290082_42.49_-72.58_" + year + ".csv"), nrows = 1)
    metadata = metadata.to_dict(orient='records')[0]

    metadata['latitude'] = metadata['Latitude']
    metadata['longitude'] = metadata['Longitude']
    metadata['TZ'] = metadata['Time Zone']
    metadata['altitude'] = metadata['Elevation']
    metadata['state'] = metadata['State']
    metadata['country'] = metadata['Country']
    metadata['county'] = '-'
    
    if (month == "season"):
        starttime = '04_30_2021'
        endtime = '09_30_2021'
    elif ((isinstance(month , int)) & (month > 1) & (month < 13)):
        starttime = ("0" if month < 10 else "") + str(month-1) + "_" + str(calendar.monthrange(2021, month-1)[1]) + "_2021"
        endtime = ("0" if month < 10 else "") + str(month) + "_" + str(calendar.monthrange(2021, month)[1]) + "_2021"
    else: raise Exception("Invalid month option")

    df_weather = pd.read_csv(os.path.join(datafolder, "1290082_42.49_-72.58_" + year + ".csv"),
                             skiprows = 2, usecols = ["Surface Albedo",
                                                     "GHI",
                                                       "DHI",
                                                       "DNI",
                                                       "Pressure",
                                                       "Wind Speed",
                                                       "Wind Direction",
                                                       "Temperature",
                                                       "Dew Point"
                                                     ])
    # keep dates in 2021 due to bifacial radiance coercion
    # pd.to_datetime('2021-06-21 12:0:0 -5'))  # Make this timezone aware, use -5 for EST.
    idx = pd.DatetimeIndex(pd.date_range('2021-01-01', '2021-12-31 23:00:00', freq = "H"))
    df_weather = df_weather.set_index(idx)

    df_weather = df_weather.rename(columns={"Surface Albedo": "surface_albedo",
                               "GHI": "ghi",
                               "DHI": "dhi",
                               "DNI": "dni",
                               "Pressure": "surface_pressure",
                               "Wind Speed": "wind_speed",
                               "Wind Direction": "wind_direction",
                               "Temperature": "air_temperature",
                               "Dew Point": "dew_point",
                              })

    demo = br.RadianceObj(simulationname,path = testfolder)  
    demo.setGround(albedo) 

    # Specifiying growth season May to Oct.
    # keep dates in 2021 due to bifacial radiance coercion
    metData = demo.readWeatherData(metadata, df_weather, starttime=starttime, endtime=endtime)

    module_1=demo.makeModule(name='mod1',x=x,y=y,numpanels=numpanels, 
                               xgap=xgap_1, tubeParams=tubeParams)

    module_2=demo.makeModule(name='mod2',x=x,y=y,numpanels=numpanels, 
                               xgap=xgap_2, tubeParams=tubeParams)

    module_3=demo.makeModule(name='mod3',x=x,y=y,numpanels=numpanels, 
                               xgap=xgap_3, tubeParams=tubeParams)

    module_4=demo.makeModule(name='mod4',x=x,y=y,numpanels=numpanels, 
                               xgap=xgap_4, tubeParams=tubeParams)  

    sceneDict_1 = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods_1, 'nRows': nRows}
    sceneDict_2 = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods_2, 'nRows': nRows, 
                    'originx': xcoord_2} 
    sceneDict_3 = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods_3, 'nRows': nRows, 
                   'originx': xcoord_3} 
    sceneDict_4 = {'tilt':tilt,'pitch': pitch,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods_4, 'nRows': nRows, 
                   'originx': xcoord_4} 

    scene_1 = demo.makeScene(module=module_1, sceneDict=sceneDict_1) 
    scene_2 = demo.makeScene(module=module_2, sceneDict=sceneDict_2, append=True) 
    scene_3 = demo.makeScene(module=module_3, sceneDict=sceneDict_3, append=True)
    scene_4 = demo.makeScene(module=module_4, sceneDict=sceneDict_4, append=True)

    # define a single custom tube for the entire Length of the row, rather than individual partial tubes for each scene
    name='Tube_row1'
    text='! genbox Metal_Aluminum_Anodized torquetube_row1 {} 0.2 0.3 | xform -t {} -0.1 -0.3 | xform -t 0 0 {}'.format(
                                                        -torquetubelength, torquetube_xcoord, hub_height-0.1)
    customObject = demo.makeCustomObject(name,text)
    scene_1.appendtoScene(customObject=customObject) # Previously had text="!xform -rz 0"

    name='Tube_row2'
    text='! genbox Metal_Aluminum_Anodized torquetube_row2 {} 0.2 0.3 | xform -t {} -0.1 -0.3 | xform -t 0 {} {}'.format(
                                                -torquetubelength, torquetube_xcoord, pitch, hub_height-0.1)
    customObject = demo.makeCustomObject(name,text)
    scene_1.appendtoScene(customObject=customObject) #, text="!xform -rz 0")

    name='Pile'

    # A surface of revolution of height z(t) = 4.2*t and radius 0.15. Composed of 32 segments (not sure why) 
    text= '! genrev Metal_Grey tube1row1 t*{} 0.15 32 | xform -t {} {} 0'.format(hub_height-0.1, pile1x, 0)
    text += '\r\n! genrev Metal_Grey tube1row2 t*{} 0.15 32 | xform -t {} {} 0'.format(hub_height-0.1, pile1x, pitch + 0)

    for i in range (1, n_piles):
        text += '\r\n! genrev Metal_Grey tube{}row1 t*{} 0.15 32 | xform -t {} {} 0'.format(i+1, hub_height-0.1, pile1x+pilesep*i, 0)
        text += '\r\n! genrev Metal_Grey tube{}row2 t*{} 0.15 32 | xform -t {} {} 0'.format(i+1, hub_height-0.1, pile1x+pilesep*i, pitch + 0)

    customObject = demo.makeCustomObject(name,text)
    scene_1.appendtoScene(customObject=customObject) #, text="!xform -rz 0")
    
    return {"demo": demo, "metData": metData, "scene_1": scene_1}


setup = make_radobj_by_timeperiod(year)
demo = setup["demo"]
metData = setup["metData"]
scene_1 = setup["scene_1"]

# ## Interim results checking

# +
# For Checking:
#demo.getfilelist()    
# -

# Final analysis will be completed with gencumsky
# Currently using gendaylit for visualization purposes only, because gencumsky oct file is saturated
demo.gendaylit(timeindex=6)  

octfile = demo.makeOct()

# +
## Comment the ! line below to run rvu from the Jupyter notebook instead of your terminal.
## Simulation will stop until you close the rvu window
# View point
# -vp <x y z>

# View direction 
# -vd 0 0.7 -0.7

# Full view
# # !rvu -vf views/front.vp -e .01 -vp -9 -37 30 -vd 0 0.7 -0.5 -pe 0.01 UMass.oct

# +
# Zoom
# # !rvu -vf views/front.vp -e .01 -vp 8 -14 6 -vd -0.3 0.7 -0.2 -pe 0.01 UMass.oct 

# +
# an attempt at picture version
# # !rpict -vf views/front.vp -vp 14 4 6 -vd 0 0.7 -0.2 UMass.oct > test.hdr

# +
# Zoom from back
# # !rvu -vf views/front.vp -e .01 -vp 0 10 5 -vd 0.1 -0.7 -0.2 -pe 0.01 UMass.oct 
# -

# genCumSky needs to be created before the octfile

demo.genCumSky()  

octfile = demo.makeOct()

# ## Analysis set up

# We need to give some value for sensorsx/y, so the format for the dict is correct, but then we will hardcode over it

analysis = br.AnalysisObj(octfile, demo.name)  
# Sensor numbers will be overwritten, Mod wanted helps Define the results file name
frontscan, backscan = analysis.moduleAnalysis(scene_1, sensorsy=28, sensorsx = 4, modWanted = 2)


groundscan = frontscan

# We're trying to do 10 sensors across the 60 ft. x area, 
# 3 sensors up the 3 m y axis, and we can't seem to get x to cycle by 10 and y to cycle by 3 at the same time.
# They're either both cycling by 3 or both by 10.

groundscan['xstart'] = xstart
groundscan['ystart'] = ystart
groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
# These hardcode over sensorx and sensorsy
groundscan['Nx'] = sensorsx
groundscan['Ny'] = sensorsy
groundscan['xinc'] = 0 
groundscan['yinc'] = yinc
groundscan['zinc'] = 0   # no tilt necessary. 
groundscan['sx_xinc'] = sx_xinc
groundscan['sx_yinc'] = 0
groundscan['orient'] = '0 0 -1' 
groundscan

# Two minutes
start = time.time()
analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan);  # compare the back vs front irradiance  
end = time.time()
print(end - start)

# Results are insolation -- Wh/m2 over the season

# <a id='step3'></a>

# ## 3. Analyse and MAP the Ground Irradiance

# Read all the files generated into one dataframe

resultsDF = br.load.read1Result(os.path.join(testfolder, "results", "irr_UMass_groundscan_Row1_Module2_Front.csv"))

resultsDF['HourlyAvgWm2'] = resultsDF['Wm2Front'] / len(metData.solpos)
resultsDF['AvgkWhm2day'] = resultsDF['Wm2Front'] / len(metData.solpos.groupby(metData.solpos.index.date).size())/1000

# Creating a new dataframe where  each element in the front irradiance list is a column. Also transpose and reverse so it looks like a top-down view of the ground. (I transposed in the column/index selection)

reversedDF = pd.pivot(resultsDF, columns = 'x', index ='y',  values = 'AvgkWhm2day').iloc[::-1]

# ## Heat map

sns.set(rc={'figure.figsize':(11.7,8.27)})

# Plot
ax = sns.heatmap(reversedDF)
# ax.set_yticks([])
# ax.set_xticks([])
ax.set_ylabel('')  
ax.set_xlabel('')  
print('')

# ## bar chart

bed_averages = pd.DataFrame(0.0, bed_x_limits.index, ['AvgkWhm2day'])
for (spc, orient) in midx:
    bed_data = reversedDF.loc[(reversedDF.index > bed_y_limits['ymin']) &
                                               (reversedDF.index  <= bed_y_limits['ymax']),
                                              (reversedDF.columns >= bed_x_limits.loc[(spc, orient), "xmin"]) &
                                               (reversedDF.columns <= bed_x_limits.loc[(spc, orient), "xmax"])]
    # Screen for points under the posts: replace any value less than 50 W/m2/hour or 0.1 kW/m2/day with NaN 
    bed_averages.loc[(spc, orient), 'AvgkWhm2day'] = bed_data.mask(bed_data < 0.1).mean(axis=None)
bed_averages.loc[("Control", "Center"), 'AvgkWhm2day'] = reversedDF.loc[(reversedDF.index > pitch/2-0.5) &
                                               (reversedDF.index  <= pitch/2+0.5),
                                              (reversedDF.columns >= -0.5) &
                                               (reversedDF.columns <= 0.5)].mean(axis=None)

# reformat as needed for a Plotting
bed_averages = bed_averages.unstack().droplevel(0, axis = 1)[['West', 'Center', 'East']]

bed_averages

# +
sns.set(rc={'figure.figsize':(6,5)})

# fig, ax = plt.subplots()
ctrl_height = bed_averages.loc["Control", "Center"]

ax = bed_averages.plot(kind='bar')
ax.set_ylabel("Average Daily Test Plot Insolation (kWh/m$^2$)")
ax.set_xlabel("Inter-Panel Spacing (ft)")
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0),
          ncol=3, fancybox=True, shadow=True)
ax.set_ylim([0, 6])
ax.set_title(year + " growing season")
scaling = [1.01, 1.03, 1.05, 1.05,
           1.01, 1.01, 1.05, 1.06,
           1.01, 1.01, 1.01, 1.01, 1.01]
for sc, p in zip(scaling, [ax.patches[i] for i in list(chain(range(4), range(5,14)))]): # skip the empty control bars
    ax.annotate(f'{p.get_height()/ctrl_height:.0%}', (p.get_x() * 1.005, p.get_height() * sc), rotation=45, size = 'small')
fig1 = plt.gcf()
# -

fig1.savefig(str(results_dir + "/" + year + "_daily_insolation"), dpi = 200, bbox_inches="tight")

# percent difference Between beds with minimum and maximum seasonal insolation
(bed_averages.max(axis=1, numeric_only=True) - bed_averages.min(axis=1, numeric_only=True))/((bed_averages.max(axis=1, numeric_only=True) + bed_averages.min(axis=1, numeric_only=True))/2)*100

# ### similar version, but for each month

# +
# Set up a df to save results for available insolation by month/year for control
# insol_avail_kWhm2day = pd.DataFrame(data=0.0, index=list(range(5, 10)), columns = ["2016", "2017", "2018"])
# -

bed_averages = pd.DataFrame(0.0,
                            pd.MultiIndex.from_product([[5, 6, 7, 8, 9], [2, 3, 4, 5], ["West", "Center", "East"]])
                            , ['AvgkWhm2day'])

for month in range(5, 10):
    setup = make_radobj_by_timeperiod(year, month = month)
    demo = setup["demo"]
    metData = setup["metData"]
    scene_1 = setup["scene_1"]
    
    demo.genCumSky() 
    octfile = demo.makeOct()
    analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan);  # compare the back vs front irradiance  
    
    resultsDF = br.load.read1Result(os.path.join(testfolder, "results", "irr_UMass_groundscan_Row1_Module2_Front.csv"))
    resultsDF['AvgkWhm2day'] = resultsDF['Wm2Front'] / len(metData.solpos.groupby(metData.solpos.index.date).size())/1000
    reversedDF = pd.pivot(resultsDF, columns = 'x', index ='y',  values = 'AvgkWhm2day').iloc[::-1]
    
    # gather 
    for (spc, orient) in midx:
        bed_data = reversedDF.loc[(reversedDF.index > bed_y_limits['ymin']) &
                                                   (reversedDF.index  <= bed_y_limits['ymax']),
                                                  (reversedDF.columns >= bed_x_limits.loc[(spc, orient), "xmin"]) &
                                                   (reversedDF.columns <= bed_x_limits.loc[(spc, orient), "xmax"])]
        # Screen for points under the posts: replace any value less than 50 W/m2/hour or 0.1 kW/m2/day with NaN 
        bed_averages.loc[(month, spc, orient), 'AvgkWhm2day'] = bed_data.mask(bed_data < 0.1).mean(axis=None)
    bed_averages.loc[(month, "Control", "Center"), 'AvgkWhm2day'] = reversedDF.loc[(reversedDF.index > pitch/2-0.5) &
                                               (reversedDF.index  <= pitch/2+0.5),
                                              (reversedDF.columns >= -0.5) &
                                               (reversedDF.columns <= 0.5)].mean(axis=None)
    # Save the available insolation for the meteorological data graph
    insol_avail_kWhm2day.loc[month, year] = bed_averages.loc[(month, "Control", "Center"), 'AvgkWhm2day']

insol_avail_kWhm2day.to_csv(str(results_dir + "/Monthly_Available_insolation_kWhm2day.csv"))

bed_averages_month = bed_averages.reset_index(names = ["Month", "Spacing", ""]) # no name for orientation to keep out of legend

sns.set(rc={'figure.figsize':(12,4)})
g = sns.catplot(data=bed_averages_month, 
                x = "Spacing", y = "AvgkWhm2day", hue='', col='Month', kind='bar')
g.set_xlabels("Inter-Panel Spacing (ft)")
g.set_ylabels("Average Daily Test Plot Insolation (kWh/m$^2$)")
axes = g.axes.flatten()
axes[0].set_title("May")
axes[1].set_title("June")
axes[2].set_title("July")
axes[3].set_title("August")
axes[4].set_title("September")
fig_bar_month = plt.gcf()

fig_bar_month.savefig(str(results_dir + "/" + year + "_daily_insolation_by_month"), dpi = 200, bbox_inches="tight")

sns.set(rc={'figure.figsize':(12,4)})
g = sns.catplot(data=bed_averages_month, 
                x = 'Month', y = "AvgkWhm2day", hue='', col="Spacing", kind='bar')
g.set_xlabels("Month")
g.set_ylabels("Average Daily Test Plot Insolation (kWh/m$^2$)")
axes = g.axes.flatten()



# ## Daily profile analysis, sunny day

# Clear Days in 2017:
# - May: 5-19
# - June: 6-15
# - July: 7-5 or 7-26
# - Sept: 9-12
#
# Partly cloudy (max 500):
# - 7-22
#
# Very overcast (max 200):
# - 7-24

analysis_date ='2021-07-26'

# prep Data frame to hold results
day_idx = [i for i in range(len(metData.datetime)) if metData.datetime[i].date() == pd.to_datetime(analysis_date).date()]
midx = pd.MultiIndex.from_product([metData.datetime[day_idx[0]:(day_idx[-1]+1)],
                                   [2, 3, 4, 5],
                                   ["West", "Center", "East"]])
hourly_bed_averages = pd.DataFrame(0.0, midx, ['Wm2Front'])

# +
start = time.time()

for didx in day_idx:
    demo.gendaylit(timeindex=didx)
    octfile = demo.makeOct()
    analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan);
    resultsDF = br.load.read1Result(os.path.join(testfolder, "results", "irr_UMass_groundscan_Row1_Module2_Front.csv"))
    reversedDF = pd.pivot(resultsDF, columns = 'x', index ='y',  values = 'Wm2Front').iloc[::-1]
    for (spc, orient) in hourly_bed_averages.xs(metData.datetime[didx], level=0).index:
        bed_data = reversedDF.loc[(reversedDF.index > bed_y_limits['ymin']) &
                                                   (reversedDF.index  <= bed_y_limits['ymax']),
                                                  (reversedDF.columns >= bed_x_limits.loc[(spc, orient), "xmin"]) &
                                                   (reversedDF.columns <= bed_x_limits.loc[(spc, orient), "xmax"])]
        # Screen for points under the posts: replace any value less than 50 with NaN 
        hourly_bed_averages.loc[(metData.datetime[didx], spc, orient), 'Wm2Front'] = bed_data.mask(bed_data < 50).mean(axis=None)
    # add in the control plot
    hourly_bed_averages.loc[(metData.datetime[didx], "Control", "Center"), 'Wm2Front'] = reversedDF.loc[(reversedDF.index > pitch/2-0.5) &
                                               (reversedDF.index  <= pitch/2+0.5),
                                              (reversedDF.columns >= -0.5) &
                                               (reversedDF.columns <= 0.5)].mean(axis=None)
    
end = time.time()
print(end - start)
# -

test = hourly_bed_averages.reset_index(level=[0,2], names = ["Time of Day", "Spacing", "Bed"])

test.to_csv(str(results_dir + "/Hourly_" + analysis_date + ".csv"))

# remove NaN at sunrise/sunset
test = test.dropna()

# +
sns.set(rc={'figure.figsize':(8,6)})
fig, axes = plt.subplots(2, 3,  sharey=True)
fig.tight_layout()

hue_order = ['West', 'Center', 'East']
dashes = {'East': (4, 1.5), 'West': (1, 1), 'Center': ''} # customize line types so center is solid

ft_2 = test.loc[2]
sns.lineplot(ax=axes[0,0], x=ft_2["Time of Day"],y=ft_2["Wm2Front"], hue=ft_2["Bed"], style=ft_2["Bed"],
             hue_order=hue_order, style_order=hue_order, dashes = dashes)

ft_3 = test.loc[3]
sns.lineplot(ax=axes[0,1], x=ft_3["Time of Day"],y=ft_3["Wm2Front"], hue=ft_3["Bed"], style=ft_3["Bed"],
             hue_order=hue_order, style_order=hue_order, dashes = dashes)

ctrl = test.loc["Control"]
sns.lineplot(ax=axes[0,2], x=ctrl["Time of Day"],y=ctrl["Wm2Front"], hue=ctrl["Bed"], hue_order=hue_order)

ft_4 = test.loc[4]
sns.lineplot(ax=axes[1,0], x=ft_4["Time of Day"],y=ft_4["Wm2Front"], hue=ft_4["Bed"], style=ft_4["Bed"],
             hue_order=hue_order, style_order=hue_order, dashes = dashes)

ft_5 = test.loc[5]
sns.lineplot(ax=axes[1,1], x=ft_5["Time of Day"],y=ft_5["Wm2Front"], hue=ft_5["Bed"], style=ft_5["Bed"],
             hue_order=hue_order, style_order=hue_order, dashes = dashes)

handles, labels = axes[1,1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98),
#           ncol=3, fancybox=True, shadow=True)

date_form = mdates.DateFormatter("%H")

for spacing, ax in zip([2, 3, "Control", 4, 5], axes.ravel()):
    # chart formatting
    # ax.set_title(str(spacing) + (" ft" if type(spacing) == int else ""), fontweight="bold")
    ax.text(x=test.loc[spacing, "Time of Day"].iloc[0], y=910,
            s=str(spacing) + (" ft" if type(spacing) == int else ""),
            weight="bold", fontsize=13, bbox = dict(boxstyle='square,pad=0.15', facecolor='white', alpha=0.72))
    ax.get_legend().remove()
    ax.xaxis.set_major_formatter(date_form)
    # ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_ylim([0, 1000])
    
# fig.text(0.5, 0.0, 'Hour of Day', ha='center')
fig.text(-0.02, 0.5, "Average Hourly Irradiance ($W/m^2$)", va='center', rotation='vertical')    
fig.delaxes(axes[1,2])
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.68, 0.48),
          fancybox=True)

fig2 = plt.gcf()
plt.show()
# -

fig2.savefig(str(results_dir + "/Hourly_" + analysis_date), dpi = 200, bbox_inches="tight")


