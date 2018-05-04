%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from salishsea_tools import (nc_tools, gsw_calls, geo_tools, viz_tools)
import seabird
import cmocean as cmo

NEP_aug = nc.Dataset('/home/ssahu/saurav/NEP36_T_S_Spice_aug.nc')


sal_aug = NEP_aug.variables['vosaline']
temp_aug = NEP_aug.variables['votemper']
spic_aug = NEP_aug.variables['spiciness']
rho_aug = NEP_aug.variables['density']

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht']


NEP_jul = nc.Dataset('/home/ssahu/saurav/NEP36_T_S_Spice_july.nc')


sal_jul  = NEP_jul.variables['vosaline']
temp_jul = NEP_jul.variables['votemper']
spic_jul = NEP_jul.variables['spiciness']
rho_jul = NEP_jul.variables['density']

y_wcvi_slice = np.arange(230,350)
x_wcvi_slice = np.arange(550,650)

dens_cont = np.arange(25.,27.,0.25/8.)
tol = 0.05

spic_iso = np.empty((sal_jul.shape[0],dens_cont.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))

for t in np.arange(sal_jul.shape[0]):
    for iter in np.arange(dens_cont.shape[0]):
        for j in np.arange(y_wcvi_slice.shape[0]):
            for i in np.arange(x_wcvi_slice.shape[0]):
                V = rho_jul[t,:,j,i]-1000
                ind = []
                ind = np.where((V>dens_cont[iter]-tol)&(V<dens_cont[iter]+tol))
                if ind != []:
                    spic_iso[t,iter,j,i] = spic_jul[t,ind[0],j,i]
                else:
                    continue
                    

                    
                    
                    

    
    



