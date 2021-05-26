#!/usr/bin/env python

import pandas as pd
import numpy as np 
from scipy import interpolate
import matplotlib.pyplot as plt 

df = pd.read_csv('data/nicer_vignetting_curve_1p4867keV.csv',
	names=['angle','area'])
print(df)

fig = plt.figure(figsize=(5,4),tight_layout=True)
plt.plot(df['angle'],df['area'],'-')
plt.xlim(0,20)
plt.ylim(0,50)
plt.xlabel('Off-axis angle (arcmin)')
plt.ylabel('Effective area (cm2)')


f_vig = interpolate.interp1d(df['angle'],df['area'],
	kind="quadratic",fill_value=(46.4,0.0),bounds_error=False) 

x = np.linspace(0,20,100)
y = f_vig(x)

plt.plot(x,y,'-')
fig.savefig('nicer_vignetting_curve_1p4867keV.pdf')

