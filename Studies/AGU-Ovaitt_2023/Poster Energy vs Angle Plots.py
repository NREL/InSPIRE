#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from pathlib import Path

testfolder = str(Path().resolve().parent / 'TEMP' /  'Agu2023Ovaitt')

if not os.path.exists(testfolder):
    os.makedirs(testfolder)
    
print ("Your simulation will be stored in %s" % testfolder)


# In[2]:


import bifacial_radiance as br
import numpy as np
import pandas as pd
import bifacialvf
import pvlib
import matplotlib.pyplot as plt
import pickle


# In[3]:


# This information helps with debugging and getting support :)
import sys, platform
print("Working on a ", platform.system(), platform.release())
print("Python version ", sys.version)
print("Pandas version ", pd.__version__)
print("bifacial_radiance version ", br.__version__)
print("bifacialvf version ", bifacialvf.__version__)
print("pvlib version ", pvlib.__version__)


# In[4]:


plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = (10, 8)


# In[5]:


downloadData = False
if downloadData == True:
    NREL_API_KEY = None  # <-- please set your NREL API key here

    # note you must use "quotes" around your key, for example:
    # NREL_API_KEY = 'DEMO_KEY'  # single or double both work fine

    # during the live tutorial, we've stored a dedicated key on our server
    if NREL_API_KEY is None:
        try:
            NREL_API_KEY = os.environ['NREL_API_KEY']  # get dedicated key for tutorial from servier
        except KeyError:
            NREL_API_KEY = 'DEMO_KEY'  # OK for this demo, but better to get your own key

    df_tmy, metadata = pvlib.iotools.get_psm3(
        latitude=37.3022, longitude=-120.4830,
        api_key=NREL_API_KEY,
        email='silvana.ovaitt@nrel.gov',  # <-- any email works here fine
        names='tmy', 
        map_variables=True)
    
    df_tmy['Year'] = 2023
    df_tmy.index = pd.to_datetime(df_tmy[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    
    # saving
    df_tmy.to_pickle(os.path.join(testfolder, 'df_tmy.pkl'))
    metadatafile = os.path.join(testfolder, 'metadata.pkl')
    with open(metadatafile, 'wb') as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
else:
    df_tmy = pd.read_pickle(os.path.join(testfolder, 'df_tmy.pkl'))
    metadatafile = os.path.join(testfolder, 'metadata.pkl')
    with open(metadatafile, 'rb') as handle:
        metadata = pickle.load(handle)


# In[6]:


monthly = df_tmy['ghi'].resample('M').sum()
monthly.plot.bar()
plt.ylabel('Monthly GHI [W h/m^2]');

'''
fig_eDemands, ax0 = plt.subplots()#1,1,figsize=(8,6), sharey=True, 
                                    #  gridspec_kw={'wspace': 2, 'width_ratios': [1]})
ax0.bar(monthly.index, monthly.values)
ax0.set_xticklabels(labels=['Feb','Apr', 'Jun', 'Aug', 'Oct', 'Dec'], rotation=0,fontsize=14)
'''


# In[8]:


df_tmy = df_tmy.tz_localize('Etc/GMT+8')


# In[9]:


location = pvlib.location.Location(latitude=metadata['latitude'],
                                   longitude=metadata['longitude'])

solar_position = location.get_solarposition(times = df_tmy.index)

tracker_data = pvlib.tracking.singleaxis(solar_position['apparent_zenith'],
                                         solar_position['azimuth'],
                                         axis_azimuth=180,
                                         backtrack=False,
                                         max_angle=60)

tilt = tracker_data['surface_tilt'].fillna(0)
azimuth = tracker_data['surface_azimuth'].fillna(0)

df_tmy['zenith'] = solar_position['zenith']
df_tmy['solar_azimuth'] = solar_position['azimuth']

df_tmy['true_tilt'] = tilt
df_tmy['true_azimuth'] = azimuth

tracker_data = pvlib.tracking.singleaxis(solar_position['apparent_zenith'],
                                         solar_position['azimuth'],
                                         axis_azimuth=180,
                                         backtrack=True,
                                         max_angle=60)

tilt = tracker_data['surface_tilt'].fillna(0)
azimuth = tracker_data['surface_azimuth'].fillna(0)

df_tmy['backtracking_tilt'] = tilt
df_tmy['backtracking_azimuth'] = azimuth


df_poa_tracker = pvlib.irradiance.get_total_irradiance(surface_tilt=tilt,
                                                       surface_azimuth=azimuth,
                                                       dni=df_tmy['dni'],
                                                       ghi=df_tmy['ghi'],
                                                       dhi=df_tmy['dhi'],
                                                       solar_zenith=solar_position['apparent_zenith'],
                                                       solar_azimuth=solar_position['azimuth'])

tracker_poa = df_poa_tracker['poa_global']

parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
cell_temperature = pvlib.temperature.sapm_cell(tracker_poa,
                                               df_tmy['temp_air'],
                                               df_tmy['wind_speed'],
                                               **parameters)

gamma_pdc = -0.004  # divide by 100 to go from %/°C to 1/°C
nameplate = 1e3
array_power = pvlib.pvsystem.pvwatts_dc(tracker_poa, cell_temperature, nameplate, gamma_pdc)

df_tmy['backtracking_Power'] = array_power

df_tmy['backtracking_POA'] = tracker_poa


# In[10]:


start_date = pd.to_datetime('2023-06-21 00:30:00-08:00')
end_date = pd.to_datetime('2023-06-21 23:30:00-08:00')
foo = df_tmy[(df_tmy.index>start_date) & (df_tmy.index<end_date)]
foo = foo[foo.ghi > 10]
foo


# In[11]:


import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%m-%d')
plt.rcParams.update({'font.size': 20})
plt.rcParams['figure.figsize'] = (10, 8)


# In[12]:


fig, ax0 = plt.subplots(1,1,figsize=(4,6))#, sharey=True, 
                                    #  gridspec_kw={'wspace': 2, 'width_ratios': [1]})

start_date = pd.to_datetime('2023-05-15 00:30:00-08:00')
end_date = pd.to_datetime('2023-08-15 23:30:00-08:00')
foo = df_tmy[(df_tmy.index>start_date) & (df_tmy.index<end_date)]
#foo = foo[foo.ghi > 10]
ax0.plot(foo.between_time('14:30','16:30')['zenith'])
#ax0.set_xticks(['a','b', 'c'])
fig.autofmt_xdate(rotation=45)
ax0.xaxis.set_major_formatter(myFmt)
ax0.set_ylabel('Sun zenith angle')
ax0.set_xlabel('Month')
##ax0.set_xticklabels(rotation=90)
#ax0.set_xticklabels(labels=['May 15','Jun', 'Jul', 'Aug', 'Sept', 'Oct'], rotation=0,fontsize=14)


# In[13]:


start_date = pd.to_datetime('2023-05-15 00:30:00-08:00')
end_date = pd.to_datetime('2023-08-15 23:30:00-08:00')
foo = df_tmy[(df_tmy.index>start_date) & (df_tmy.index<end_date)]
foo = foo.between_time('14:30','16:30')


# In[69]:


energy_mod = pd.DataFrame()
poa_mod = pd.DataFrame()

tilts=[0.0,2.0,4.0,6.0,8.0,10.0,15.0,20.0]

for ii in range(0,len(tilts)):
    foo['new_tilt'] = foo['backtracking_tilt'].copy().values-tilts[ii]
    foo.loc[foo['new_tilt']<0, 'new_tilt'] = 0.0
    foo.loc[foo['new_tilt']>60, 'new_tilt'] = 60.0

    df_poa_tracker = pvlib.irradiance.get_total_irradiance(surface_tilt=foo['new_tilt'],
                                                           surface_azimuth=foo['backtracking_azimuth'],
                                                           dni=foo['dni'],
                                                           ghi=foo['ghi'],
                                                           dhi=foo['dhi'],
                                                           solar_zenith=foo['zenith'],
                                                           solar_azimuth=foo['solar_azimuth'])

    tracker_poa = df_poa_tracker['poa_global']

    parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
    cell_temperature = pvlib.temperature.sapm_cell(tracker_poa,
                                                   foo['temp_air'],
                                                   foo['wind_speed'],
                                                   **parameters)

    gamma_pdc = -0.004  # divide by 100 to go from %/°C to 1/°C
    nameplate = 1e3
    array_power = pvlib.pvsystem.pvwatts_dc(tracker_poa, cell_temperature, nameplate, gamma_pdc)
    energy_mod[str(tilts[ii])] = 100-array_power*100/foo['backtracking_Power']
    poa_mod[str(tilts[ii])] = foo['backtracking_POA']-tracker_poa


# In[70]:


monthly = energy_mod.resample('M').mean()
monthly.plot.bar()
plt.ylabel('% Monthly Power Loss');
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))


# In[26]:


foo.keys()


# ## PVLIB PVFACTORS

# In[52]:


start_date = pd.to_datetime('2023-05-15 00:30:00-08:00')
end_date = pd.to_datetime('2023-08-15 23:30:00-08:00')
foo = df_tmy[(df_tmy.index>start_date) & (df_tmy.index<end_date)]
foo = foo.between_time('14:30','16:30')
foo


# In[98]:


# example array geometry
pvrow_height = 1.5
pvrow_width = 2
pitch = 5.7
gcr = pvrow_width / pitch
axis_azimuth = 180
albedo = 0.2
parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
bifaciality = 0.7
gamma_pdc = -0.004  # divide by 100 to go from %/°C to 1/°C
nameplate = 1e3


# In[99]:


# Bifi BASELINE
irrad = pvlib.bifacial.pvfactors_timeseries(
    solar_azimuth=foo['solar_azimuth'],
    solar_zenith=foo['zenith'],
    surface_azimuth=foo['backtracking_azimuth'],  # south-facing array
    surface_tilt=foo['backtracking_tilt'],
    axis_azimuth=0, # 90 degrees off from surface_azimuth.  270 is ok too
    timestamps=foo.index,
    dni=foo['dni'],
    dhi=foo['dhi'],
    gcr=gcr,
    pvrow_height=pvrow_height,
    pvrow_width=pvrow_width,
    albedo=albedo,
    n_pvrows=3,
    index_observed_pvrow=1
)

irrad = pd.concat(irrad, axis=1)


effective_irrad_bifi = irrad['total_abs_front'] + (irrad['total_abs_back']
                                                   * bifaciality)
effective_irrad_mono = irrad['total_abs_front']

# bifi
cell_temperature = pvlib.temperature.sapm_cell(effective_irrad_bifi,
                                               foo['temp_air'],
                                               foo['wind_speed'],
                                               **parameters)
array_power_bifi = pvlib.pvsystem.pvwatts_dc(effective_irrad_bifi, cell_temperature, nameplate, gamma_pdc)

foo['backtracking_Power_bifi'] = array_power_bifi

# bifi
cell_temperature = pvlib.temperature.sapm_cell(effective_irrad_mono,
                                               foo['temp_air'],
                                               foo['wind_speed'],
                                               **parameters)
array_power_mono = pvlib.pvsystem.pvwatts_dc(effective_irrad_mono, cell_temperature, nameplate, gamma_pdc)

foo['backtracking_Power_mono'] = array_power_mono


# In[100]:


energy_bifi = pd.DataFrame()
energy_mono = pd.DataFrame()

tilts=[0.0,2.0,4.0,6.0,8.0,10.0,15.0,20.0, 40.0]

for ii in range(0,len(tilts)):
    foo['new_tilt'] = foo['backtracking_tilt'].copy().values-tilts[ii]
    foo.loc[foo['new_tilt']<0, 'new_tilt'] = 0.0
    foo.loc[foo['new_tilt']>60, 'new_tilt'] = 60.0
   
    irrad = pvlib.bifacial.pvfactors.pvfactors_timeseries(
        solar_azimuth=foo['solar_azimuth'],
        solar_zenith=foo['zenith'],
        surface_azimuth=foo['backtracking_azimuth'],  # south-facing array
        surface_tilt=foo['new_tilt'],
        axis_azimuth=0, # 90 degrees off from surface_azimuth.  270 is ok too
        timestamps=foo.index,
        dni=foo['dni'],
        dhi=foo['dhi'],
        gcr=gcr,
        pvrow_height=pvrow_height,
        pvrow_width=pvrow_width,
        albedo=albedo,
        n_pvrows=3,
        index_observed_pvrow=1
    )

    irrad = pd.concat(irrad, axis=1)


    effective_irrad_bifi = irrad['total_abs_front'] + (irrad['total_abs_back']
                                                       * bifaciality)
    effective_irrad_mono = irrad['total_abs_front']
    
    # bifi
    cell_temperature = pvlib.temperature.sapm_cell(effective_irrad_bifi,
                                                   foo['temp_air'],
                                                   foo['wind_speed'],
                                                   **parameters)
    array_power_bifi = pvlib.pvsystem.pvwatts_dc(effective_irrad_bifi, cell_temperature, nameplate, gamma_pdc)
    
    # mono
    cell_temperature = pvlib.temperature.sapm_cell(effective_irrad_mono,
                                                   foo['temp_air'],
                                                   foo['wind_speed'],
                                                   **parameters)
    array_power_mono = pvlib.pvsystem.pvwatts_dc(effective_irrad_mono, cell_temperature, nameplate, gamma_pdc)
    
    energy_bifi[str(tilts[ii])] = 100-array_power_bifi*100/(foo['backtracking_Power_bifi'])
    energy_mono[str(tilts[ii])] = 100-array_power_mono*100/(foo['backtracking_Power_mono'])


# In[101]:


monthly_bifi = energy_bifi.resample('M').mean()
monthly_bifi.plot.bar()
plt.ylabel('% Monthly Power Loss');
ax = plt.gca()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))


# In[102]:


monthly_mono = energy_mono.resample('M').mean()


# In[103]:


plt.plot(monthly_mono.iloc[3],label='Monofacial system')
plt.plot(monthly_bifi.iloc[3], label='Bifacial_system')
plt.legend()
plt.xlabel('Tracking angle offset')
plt.ylabel('Energy % loss')


# ## MERCED Raytrace 

# In[105]:


simulationname = 'tutorial_1'

# Scene Parameters:
azimuth_ang=180 # Facing south
tilt =35 # tilt.

# MakeModule Parameters
moduletype='test-module'
numpanels = 1  # AgriPV site has 3 modules along the y direction
module_x = 2 # m
module_y = 1 # m. slope we will measure
# if x > y, landscape. if x < y, portrait
xgap = 0.0#
ygap = 0.0 # Leaving 10 centimeters between modules on y direction
zgap = 0.0
sensorsy = 10  # this will give 6 sensors per module, 1 per cell

# SceneDict Parameters
pitch = 5.7 # m
albedo = 0.2  #'grass'     # ground albedo
hub_height = 1.5 # m  
nMods = 10 # six modules per row.
nRows = 5  # 3 row


# In[106]:


demo = br.RadianceObj(simulationname,path = testfolder)  
demo.setGround(albedo) 
epwfile = demo.getEPW(lat, lon) # NJ lat/lon 40.0583° N, 74.4057


# In[ ]:


# Making module, either as a black Unit or with cell-level detail.
# Suggest to use cell-level only for visualizations, and or for studying customly made agriPV modules where the cell
# gaps might be really, really wide. Most commercial panels can be approximated by the single-black surface.
detailedModule = False

if detailedModule:
    cellModule = {'numcellsx': numcellsx, 'numcellsy':numcellsy, 
                             'xcell': xcell, 'ycell': ycell, 'xcellgap': xcellgap, 'ycellgap': ycellgap}
    module=demo.makeModule(name=moduletype,numpanels=numpanels, 
                           xgap=xgap, ygap=ygap, cellModule=cellModule, tubeParams=tubeParams)
else:
    module=demo.makeModule(name=moduletype,x=module_x,y=module_y,numpanels=numpanels, 
                           xgap=xgap, ygap=ygap, tubeParams=tubeParams)


# In[ ]:


#Determine Hour to model
#Valid options: mm_dd, mm_dd_HH, mm_dd_HHMM, YYYY-mm-dd_HHMM
metdata = demo.readWeatherFile(epwfile, coerce_year=2021, starttime='2021-06-21_12', endtime='2021-06-21_13')


# In[ ]:


demo.metdata.__dict__


# In[ ]:


metdata.__dict__


# We are going to model just one single timeindex at a time.

# In[ ]:


timeindex = metdata.datetime.index(pd.to_datetime('2021-06-21 12:0:0 -5'))  # Make this timezone aware, use -5 for EST.
timeindex


# In[ ]:


demo.genCumSky()  


# In[ ]:


demo.gendaylit(timeindex=timeindex)  


# In[ ]:


sceneDict = {'tilt':tilt,'pitch': 15,'hub_height':hub_height,'azimuth':azimuth_ang, 'nMods': nMods, 'nRows': nRows}  
scene = demo.makeScene(module=moduletype, sceneDict=sceneDict) 


# In[ ]:


octfile = demo.makeOct(demo.getfilelist())


# If desired, you can view the Oct file at this point:
# 
# ***rvu -vf views\front.vp -e .01 tutorial_1.oct***

# In[ ]:


## Comment the ! line below to run rvu from the Jupyter notebook instead of your terminal.
## Simulation will stop until you close the rvu window

get_ipython().system('rvu -vf views\\front.vp -e .01 tutorial_1.oct')


# And adjust the view parameters, you should see this image.
# 
# ![AgriPV modeled step 1](images/AgriPV_step1.PNG)
# 

# ## 4. Adding different Albedo Sections
# For practicing adding custom scene elements, we will add a patch in the ground that has a different reflectivity (albedo) than the average set for the field. 
# By using this `genbox` and giving it the right size/position, we can create trees, buildings, or secondary small-area surfaces to add with sampling at specific heights but that do not shade the scene.
# 

# In[ ]:


name='Center_Patch'
patchpositionx=-14
patchpositiony=2
text='! genbox white_EPDM CenterPatch 28 12 0.001 | xform -t {} {} 0'.format(patchpositionx, patchpositiony)
customObject = demo.makeCustomObject(name,text)
demo.appendtoScene(scene.radfiles, customObject)
octfile = demo.makeOct(demo.getfilelist()) 


# In[ ]:


#!rvu -vf views\front.vp -e .01 tutorial_1.oct
#!rvu -vf views\front.vp -e .01 -pe 0.4 -vp 12 -10 3.5 -vd -0.0995 0.9950 0.0 tutorial_1.oct


# Viewing with rvu:
# 
# ![AgriPV modeled step 4](images/AgriPV_step4.PNG)
# 
# 

# ## 2. Analyse  the Ground Irradiance
# 
# Now let's do some analysis along the ground, starting from the edge of the modules. We wil select to start in the center of the array.
# 
# We are also increasign the number of points sampled accross the collector width, with the  variable **sensorsy** passed to **moduleanalysis**. We are also increasing the step between sampling points, to be able to sample in between the rows.

# In[ ]:


analysis = br.AnalysisObj(octfile, demo.name)  
frontscan, backscan = analysis.moduleAnalysis(scene, sensorsy=sensorsy, modWanted=4)


# In[ ]:


frontscan


# In[ ]:


groundscan = frontscan


# In[ ]:


groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
groundscan['zinc'] = 0   # no tilt necessary. 
groundscan['yinc'] = pitch/(sensorsy-1)   # increasing spacing so it covers all distance between rows
groundscan['orient'] = '0 0 -1' 
groundscan


# In[ ]:


analysis.analysis(octfile, simulationname+"_groundscan", groundscan, backscan)  # compare the back vs front irradiance  


# This is the result for only one 'chord' accross the ground. Let's now do a X-Y scan of the ground.

# <a id='step3'></a>

# ## 3. Analyse and MAP the Ground Irradiance

#  We will use the same technique to find the irradiance on the ground used above, but will move it along the X-axis to map from the start of one module to the next.
#  
#  We will sample around the module that is placed at the center of the field.

# ![AgriPV modeled step 4](images/spacing_between_modules.PNG)

# In[ ]:


import seaborn as sns


# In[ ]:


sensorsx = 3
startgroundsample=-module.scenex
spacingbetweensamples = module.scenex/(sensorsx-1)

for i in range (0, sensorsx): # Will map 20 points    
    frontscan, backscan = analysis.moduleAnalysis(scene, sensorsy=sensorsy)
    groundscan = frontscan
    groundscan['zstart'] = 0.05  # setting it 5 cm from the ground.
    groundscan['zinc'] = 0   # no tilt necessary. 
    groundscan['yinc'] = pitch/(sensorsy-1)   # increasing spacing so it covers all distance between rows
    groundscan['xstart'] = startgroundsample + i*spacingbetweensamples   # increasing spacing so it covers all distance between rows
    analysis.analysis(octfile, simulationname+"_groundscan_"+str(i), groundscan, backscan)  # compare the back vs front irradiance  


# Read all the files generated into one dataframe

# In[ ]:


filestarter = "irr_tutorial_1_groundscan_"

filelist = sorted(os.listdir(os.path.join(testfolder, 'results')))
prefixed = [filename for filename in filelist if filename.startswith(filestarter)]
arrayWm2Front = []
arrayWm2Back = []
arrayMatFront = []
arrayMatBack = []
filenamed = []
faillist = []

print('{} files in the directory'.format(filelist.__len__()))
print('{} groundscan files in the directory'.format(prefixed.__len__()))
i = 0  # counter to track # files loaded.

for i in range (0, len(prefixed)):
    ind = prefixed[i].split('_')

    try:
        resultsDF = br.load.read1Result(os.path.join(testfolder, 'results', prefixed[i]))
        arrayWm2Front.append(list(resultsDF['Wm2Front']))
        arrayWm2Back.append(list(resultsDF['Wm2Back']))
        arrayMatFront.append(list(resultsDF['mattype']))
        arrayMatBack.append(list(resultsDF['rearMat']))
        filenamed.append(prefixed[i])
    except:
        print(" FAILED ", i, prefixed[i])
        faillist.append(prefixed[i])

resultsdf = pd.DataFrame(list(zip(arrayWm2Front, arrayWm2Back, 
                                  arrayMatFront, arrayMatBack)),
                         columns = ['br_Wm2Front', 'br_Wm2Back', 
                                    'br_MatFront', 'br_MatBack'])
resultsdf['filename'] = filenamed


# Creating a new dataframe where  each element in the front irradiance list is a column. Also transpose and reverse so it looks like a top-down view of the ground.

# In[ ]:


df3 = pd.DataFrame(resultsdf['br_Wm2Front'].to_list())
reversed_df = df3.T.iloc[::-1]


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[ ]:


# Plot
ax = sns.heatmap(reversed_df)
ax.set_yticks([])
ax.set_xticks([])
ax.set_ylabel('')  
ax.set_xlabel('')  
print('')


# <a id='step4'></a>
