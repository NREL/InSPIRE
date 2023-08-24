# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:34:05 2023

@author: akinzer
"""

# In[0]
import os
import pandas as pd
import numpy as np
import json

# In[1]
resultsfolder = '/scratch/akinzer/AgriGrapes/'
with open("locationlist", "r") as fp:
    b = json.load(fp)


# Folders where results are and will be saved
savefolder='/scratch/akinzer/AgriGrapes/'
#posSampled = 50 #!! Modify to the number of positions sampled

ft2m = 0.3048

# In[2]
# Variables looping through in the folder title
# array hub-height
# hubheights = [1.5, 2.4] # meters
# rtrs = [5.0, 10.0]# ,6.0] # meters
# xgaps = [0.0, 1.0]# ,6.0] # meters
timestamps = np.array(range(0,8760)).astype('str')
timestamps1 = np.array(range(0,8760)).astype('str')
timestamp_length = 4
pad = '0'
for i in range(0,8760):
    timestamps[i] = (timestamp_length - len(timestamps[i]))*pad + timestamps[i]

# In[3]
# locations = np.array(b) # choose specific locations for analysis
# locations_co = [c for c in b if 'USA_CO' in c] # choose Colorado locations
# periods = np.array(['5TO5', '6TO6', '7TO7', '8TO8', '9TO9','5TO9']) # add more to match other time periods

# tilts = np.array(['-50.0', '-45.0', '-40.0', '-35.0', '-30.0', '-25.0', '-20.0', '-15.0', '-10.0', '-5.0', '-0.0', 
#                  '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '35.0', '40.0', '45.0', '50.0'])

#irr_1axis_25.0Ground_Row4_Module10_Front.csv
#ignore irr_1axis_25.0Ground_Row4_Module10_Rear.csv
#irr_1axis_20.0Module_Row4_Module10.cs
#C:\Users\akinzer\Documents\AUSTIN\AgriGrapes\Timestamp_0000\results\irr_setup2_0_Row1_Module1_Front.csv

x_all = []
y_all = []
z_all = []
Wm2Front_all = []
timestamp_all = []

# xmod_all = []
# ymod_all = []
# zmod_all = []
# rearZ_all = []
# Wm2Front_all = []
# Wm2Back_all = []

# location_all = []
# xgap_all = []
# hubheight_all = []
# rtr_all = []
# period_all = []
# tilt_all = []

errors_all = []
for ii in range(0, len(timestamps)):
    timestamp_padded = timestamps[ii]
    timestamp = timestamps1[ii]    

    filenameModule = os.path.join(resultsfolder, (
                'Timestamp_{}/results/irr_setup2_{}_Row1_Module1_Front.csv'.format(timestamp_padded, timestamp)))
    print("Working on entry {}".format(timestamp_padded))

    try:
        data = pd.read_csv(filenameModule)
        
        # Save all the values
        x_all.append(list(data['x']))
        y_all.append(list(data['y']))
        z_all.append(list(data['z']))
        Wm2Front_all.append(list(data['Wm2Front']))
    
        # Saving position and parameters for indexing
        timestamp_all.append(timestamp)                           
        
    except:
        print('*** Missing entry {}'.format(timestamp))
        errors_all.append('*** Missing entry {}'.format(timestamp))
            

savefilename = 'Results_Clifton_{}.csv'.format(timestamp)

df = pd.DataFrame(list(zip(Wm2Front_all, x_all, y_all, z_all, timestamp_all)),
                    columns=['Wm2Front','xground','yground','zground', 'timestamp'])

df.to_pickle(os.path.join(savefolder,savefilename))
df.to_csv(os.path.join(savefolder,'Results_Clifton.csv'))

# open file and read the content in a list
with open(r'/scratch/akinzer/AgriGrapes/ERRORS_compile.txt', 'w') as fp:
    fp.write('\n'.join(errors_all))


print("FINISHED")