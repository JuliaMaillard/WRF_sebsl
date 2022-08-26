#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:27:20 2022

@author: maillardj
"""

import surfacelayermodels_Maillard2022 as slm
import numpy as np
import matplotlib.pyplot as plt

#%% Test
Ta      = 273.15 - 15.
Tg      = 273.15 - 2.
LWd     = 180
Ua      = 3
dsnow   = 0.3

# create Noah - MYJ object
onelay  = slm.noahmyj()

# solve to get Ts
Ts      = onelay.solve(Ua, Ta, LWd, dsnow, Tg)
print(Ts)

# create Noah - MP object
twolay  = slm.noahmp()

# solve to get Ts and Tc
Ts, Tc  = twolay.solve(Ua, Ta, LWd, dsnow, Tg)
print(Ts, Tc)

#%% For several different wind conditions
Ta      = 273.15 - 15.
Tg      = 273.15 - 2.
LWd     = 180
dsnow   = 0.3

oMYJ    = slm.noahmyj(version='ORIG')
mMYJ    = slm.noahmyj(version='MODIF')
oMP     = slm.noahmp(version='ORIG')
mMP     = slm.noahmp(version='MODIF')

Ntest   = 100
Ua      = np.linspace(0,10,Ntest)    
Ts_omyj = np.zeros(Ntest) + np.nan
Ts_mmyj = np.zeros(Ntest) + np.nan
Ts_omp  = np.zeros(Ntest) + np.nan
Ts_mmp  = np.zeros(Ntest) + np.nan

for ii in range(Ntest):
    Ts_omyj[ii]     = oMYJ.solve(Ua[ii], Ta, LWd, dsnow, Tg)
    Ts_mmyj[ii]     = mMYJ.solve(Ua[ii], Ta, LWd, dsnow, Tg)
    Ts_omp[ii]      = oMP.solve(Ua[ii], Ta, LWd, dsnow, Tg)[0]
    Ts_mmp[ii]      = mMP.solve(Ua[ii], Ta, LWd, dsnow, Tg)[0]

fig = plt.figure(figsize=(8,5), dpi=100)
plt.plot(Ua, Ta-Ts_omyj)
plt.plot(Ua, Ta-Ts_mmyj)
plt.plot(Ua, Ta-Ts_omp)
plt.plot(Ua, Ta-Ts_mmp)
plt.plot(Ua, Ta-Ts)
plt.grid()
plt.xlim([0,10])
plt.ylim([0,13])
