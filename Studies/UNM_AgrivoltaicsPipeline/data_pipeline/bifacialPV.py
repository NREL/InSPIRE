from pathlib import Path
import os
import bifacialvf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pvlib
import sys
import platform
from bifacialvf import loadVFresults

# Create output folder
testfolder = 'TEMP'
os.makedirs(testfolder, exist_ok=True)

# Debug information
print("Working on a", platform.system(), platform.release())
print("Python version:", sys.version)
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Pvlib version:", pvlib.__version__)
print("bifacialVF version:", bifacialvf.__version__)

# Plot settings
plt.rcParams['timezone'] = 'Etc/GMT+7'
plt.rc('font', family='DejaVu Sans', weight='bold', size=22)
plt.rcParams['figure.figsize'] = (12, 5)

# Simulation parameters
writefiletitle = os.path.join(testfolder, 'Results_bifacialVF.csv')
lat = 39.7555
lon = -105.2211
tilt = 30
sazm = 180
albedo = None
module_slope = 2
clearance_height = 1.5 / module_slope
pitch = 2 / 0.35 / module_slope
rowType = "interior"
transFactor = 0
sensorsy = 12
PVfrontSurface = "glass"
PVbackSurface = "glass"
agriPV = True
tracking = False
backtrack = True
limit_angle = 50
deltastyle = 'TMY3'

# Get weather file and run simulation
TMYtoread = bifacialvf.getEPW(lat=lat, lon=lon, path=testfolder)
myTMY3, meta = bifacialvf.readInputTMY(TMYtoread)
bifacialvf.simulate(
    myTMY3, meta, writefiletitle=writefiletitle,
    tilt=tilt, sazm=sazm, pitch=pitch, clearance_height=clearance_height,
    rowType=rowType, transFactor=transFactor, sensorsy=sensorsy,
    PVfrontSurface=PVfrontSurface, PVbackSurface=PVbackSurface,
    albedo=albedo, tracking=tracking, backtrack=backtrack,
    limit_angle=limit_angle, deltastyle=deltastyle, agriPV=agriPV
)

# Load and process simulation results
data, metadata = loadVFresults(writefiletitle)
data.set_index(pd.to_datetime(data['date']), inplace=True, drop=True)
data.index = data.index.map(lambda t: t.replace(year=2021))

# Extract and convert Ground Irradiance Values
groundIrrads = data['Ground Irradiance Values'].str.strip('[]').str.split(' ', expand=True)
groundIrrads = groundIrrads.applymap(lambda x: float(x.replace('np.float64(', '').replace(')', '')) if isinstance(x, str) else x)

# Daily totals, transposed
df = groundIrrads.groupby(pd.Grouper(freq='D')).sum().T

# Convert from Wh/m² to MJ/m²
df *= 0.0036

# Save final CSV
df.to_csv('daily_sensors.csv', index=True)
print("Saved: daily_sensors.csv")
