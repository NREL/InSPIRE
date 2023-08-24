"""
# COMPILE GROUND IRRADIANCE FROM BIFACIAL_RADIANCE HPC RUN

"""


import os
import pandas as pd
import numpy as np
import json

resultsfolder = '/projects/inspire/JordanWorld/'
with open("locationlist", "r") as fp:
    b = json.load(fp)


# Folders where results are and will be saved
savefolder='/home/akinzer/JordanWorld'
#posSampled = 50 #!! Modify to the number of positions sampled

ft2m = 0.3048

# Variables looping through in the folder title
# array hub-height
hubheights = [1.5, 2.4] # meters
rtrs = [5.0, 10.0]# ,6.0] # meters
xgaps = [0.0, 1.0]# ,6.0] # meters

locations = np.array(b) # choose specific locations for analysis
locations_co = [c for c in b if 'USA_CO' in c] # choose Colorado locations
periods = np.array(['5TO5', '6TO6', '7TO7', '8TO8', '9TO9','5TO9']) # add more to match other time periods

tilts = np.array(['-50.0', '-45.0', '-40.0', '-35.0', '-30.0', '-25.0', '-20.0', '-15.0', '-10.0', '-5.0', '-0.0', 
                 '5.0', '10.0', '15.0', '20.0', '25.0', '30.0', '35.0', '40.0', '45.0', '50.0'])

#irr_1axis_25.0Ground_Row4_Module10_Front.csv
#ignore irr_1axis_25.0Ground_Row4_Module10_Rear.csv
#irr_1axis_20.0Module_Row4_Module10.cs
x_all = []
y_all = []
z_all = []
Wm2Ground_all = []

xmod_all = []
ymod_all = []
zmod_all = []
rearZ_all = []
Wm2Front_all = []
Wm2Back_all = []

location_all = []
xgap_all = []
hubheight_all = []
rtr_all = []
period_all = []
tilt_all = []

errors_all = []
for ii in range(0, len(locations)):
    location = locations[ii]
    for jj in range(0, len(hubheights)):
        hubheight = hubheights[jj]
        for kk in range(0, len(rtrs)):
            rtr = rtrs[kk]
            for ll in range(0,len(xgaps)):
                xgap = xgaps[ll]
                for mm in range(0,len(periods)):
                    period = periods[mm]

           

                    for nn in range(0,len(tilts)):
                        tilt = tilts[nn]

                        
                        # USA_CO_Golden-NREL__724666_TMY3_hh1.5_rtr10.0_xgap1.0_from_9TO9
                        filenameGround = os.path.join(resultsfolder, (
                                    '{}_hh{}_rtr{}_xgap{}_from_{}/results/irr_1axis_{}Ground_Row4_Module10_Front.csv'.format(location, 
                                        hubheight, rtr, xgap, period, tilt)))
                        
                        filenameModule = os.path.join(resultsfolder, (
                                    '{}_hh{}_rtr{}_xgap{}_from_{}/results/irr_1axis_{}Module_Row4_Module10.csv'.format(location, 
                                        hubheight, rtr, xgap, period, tilt)))
                        print("Working on entry {}_hh{}_rtr{}_xgap{}_from{}".format(location, hubheight, rtr, xgap, period))
            
                        try:
                            data = pd.read_csv(filenameGround)
                            
                            # Save all the values
                            x_all.append(list(data['x']))
                            y_all.append(list(data['y']))
                            z_all.append(list(data['z']))
                            Wm2Ground_all.append(list(data['Wm2Front']))
                        
                            data = pd.read_csv(filenameModule)
                            xmod_all.append(list(data['x']))
                            ymod_all.append(list(data['y']))
                            zmod_all.append(list(data['z']))
                            rearZ_all.append(list(data['rearZ']))
                            Wm2Front_all.append(list(data['Wm2Front']))
                            Wm2Back_all.append(list(data['Wm2Back']))
            
                            # Saving position and parameters for indexing
                            location_all.append(location)
                            hubheight_all.append(hubheight)
                            rtr_all.append(rtr)
                            period_all.append(period)
                            xgap_all.append(xgap)
                            tilt_all.append(tilt)                           
                            
                        except:
                            print('*** Missing entry {}_hh{}_rtr{}_xgap{}_from{}'.format(location, hubheight, rtr, xgap, period))
                            errors_all.append('*** Missing entry {}_hh{}_rtr{}_xgap{}_from{}'.format(location, hubheight, rtr, xgap, period))
                            

savefilename = 'Results_p2_{}_hh{}_rtr{}_xgap{}_from{}.csv'.format(location, hubheight, rtr, xgap, period)

df = pd.DataFrame(list(zip(location_all,hubheight_all, rtr_all,period_all,
                            xgap_all,tilt_all,xmod_all,ymod_all,zmod_all,rearZ_all,
                            Wm2Front_all,Wm2Back_all,x_all, y_all, z_all, Wm2Ground_all)),
                    columns=['location', 'hubheight', 'rtr', 'period', 'xgap','tilt','xmod','ymod','zmod','rearZ','Wm2Front','Wm2Back',
                                'xground','yground','zground', 'Wm2Ground'])

df.to_pickle(os.path.join(savefolder,savefilename))
df.to_csv(os.path.join(savefolder,'Results_Golden.csv'))

# open file and read the content in a list
with open(r'/home/akinzer/JordanWorld/ERRORS_compile.txt', 'w') as fp:
    fp.write('\n'.join(errors_all))


print("FINISHED")