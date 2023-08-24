# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:01:26 2023

@author: akinzer
"""

import os
import pandas as pd
import numpy as np
import json

# os.chdir('C:\\Users\\akinzer\\Documents\Technical_Assistance\Kebwezi\results')

resultsfolder = '/home/akinzer/BasicSimulations/TEMP/results'

# Folders where results are and will be saved
savefolder='/home/akinzer/BasicSimulations/TEMP/results'

positions = np.array(range(0,21))


# hubheights = [1.5, 2.4] # meters
# rtrs = [5.0, 10.0]# ,6.0] # meters
# xgaps = [0.0, 1.0]# ,6.0] # meters

# locations = np.array(b) # choose specific locations for analysis
# locations_co = [c for c in b if 'USA_CO' in c] # choose Colorado locations
# periods = np.array(['5TO5', '6TO6', '7TO7', '8TO8', '9TO9','5TO9']) # add more to match other time periods

# tilts = np.array(['-50.0', '-45.0', '-40.0', '-35.0', '-30.0', '-25.0', '-20.0', '-15.0', '-10.0', '-5.0', '-0.0', 
#                  '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '35.0', '40.0', '45.0', '50.0'])

#irr_Temple_checkerboard_groundscan_10_Row1_Module2.csv
x_all = []
y_all = []
z_all = []

rearZ_all = []
Wm2Front_all = []
Wm2Back_all = []

positions_all = []

errors_all = []

for i in positions:
    position = positions[i]
    filenameGround = os.path.join(resultsfolder, ('irr_Temple_checkerboard_groundscan_{}_Row1_Module2.csv'.format(position)))
                        
    filenameModule = os.path.join(resultsfolder, ('irr_Temple_checkerboard_groundscan_{}_Row1_Module2.csv'.format(position)))
    print("Working on entry {}".format(position))
            
    try:
        data = pd.read_csv(filenameGround)
                            
        # Save all the values
        x_all.append(list(data['x']))
        y_all.append(list(data['y']))
        z_all.append(list(data['z']))
    
        data = pd.read_csv(filenameModule)
        rearZ_all.append(list(data['rearZ']))
        Wm2Front_all.append(list(data['Wm2Front']))
        Wm2Back_all.append(list(data['Wm2Back']))

        # Saving position and parameters for indexing
        positions_all.append(position)                   
        
    except:
        print('*** Missing entry {}'.format(position))
        errors_all.append('*** Missing entry {}'.format(position))
        

savefilename = 'Results_Temple.csv'

df = pd.DataFrame(list(zip(positions_all,rearZ_all,
                            Wm2Front_all,Wm2Back_all,x_all, y_all, z_all)),
                    columns=['position','rearZ','Wm2Front','Wm2Back',
                                'xground','yground','zground'])

df.to_pickle(os.path.join(savefolder,savefilename))
df.to_csv(os.path.join(savefolder,'Results_Temple.csv'))

# open file and read the content in a list
with open(r'/home/akinzer/BasicSimulations/TEMP/results/ERRORS_compile.txt', 'w') as fp:
    fp.write('\n'.join(errors_all))


print("FINISHED")
