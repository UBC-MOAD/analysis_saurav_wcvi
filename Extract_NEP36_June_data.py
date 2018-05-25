from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import os
import glob
import fnmatch
from collections import namedtuple, OrderedDict
import scipy.io as sio
from scipy import interpolate, signal
from pyproj import Proj,transform
import sys
sys.path.append('/ocean/ssahu/CANYONS/wcvi/grid/')
from bathy_common import *
from matplotlib import path
import xarray as xr
import scipy.io as sio
import matplotlib.cm as cm
import cmocean as cmo
import matplotlib.gridspec as gridspec
from dateutil.parser import parse
from salishsea_tools import geo_tools, viz_tools, tidetools, nc_tools
import gsw

path_to_save = '/data/ssahu/NEP36_Extracted_Months/'


bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')

Z = bathy.variables['Bathymetry'][:]

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht']

lon = bathy['nav_lon'][...]
lat = bathy['nav_lat'][...]

z0 = np.ma.masked_values(Z, 0)

y_wcvi_slice = np.array(np.arange(230,350))
x_wcvi_slice = np.array(np.arange(550,650))

def tem_sal_timeseries_at_WCVI_locations(grid_scalar):#, j, i):

    temp = grid_scalar.variables['votemper'][0,:, :, :]
    sal = grid_scalar.variables['vosaline'][0,:, :, :]
    
    scalar_ts = namedtuple('scalar_ts', 'temp, sal')

    return scalar_ts(temp, sal)


print("Extracting June Data")    
    
    
temp_june = np.empty((30,50,Z.shape[0],Z.shape[1]))
sal_june = np.empty((30,50,Z.shape[0],Z.shape[1]))


i = 0
for file in sorted(glob.glob('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_201506*grid_T.nc')):

    scalar_ts = tem_sal_timeseries_at_WCVI_locations(nc.Dataset(file))
    temp_june[i,...] = scalar_ts[0]
    sal_june[i,...] = scalar_ts[1]
    i = i+1

print("Calculating the Spice for June for the WCVI subset; the rest of the locations will have empty spice")
    
pressure_loc = gsw.p_from_z(-zlevels[:],np.mean(lat))

SA_loc_jun = np.empty_like(sal_june)
CT_loc_jun = np.empty_like(sal_june)
spic_jun = np.empty_like(sal_june)
rho_jun = np.empty_like(sal_june)

for t in np.arange(sal_june.shape[0]):
    for k in np.arange(sal_june.shape[1]):
        for j in np.arange(230,350):
            for i in np.arange(550,650):
                SA_loc_jun[t,k,j,i] = gsw.SA_from_SP(sal_june[t,k,j,i], pressure_loc[k], lon[j,i], lat[j,i])
                CT_loc_jun[t,k,j,i] = gsw.CT_from_pt(sal_june[t,k,j,i], temp_june[t,k,j,i])
                spic_jun[t,k,j,i] = gsw.spiciness0(SA_loc_jun[t,k,j,i],CT_loc_jun[t,k,j,i])
                rho_jun[t,k,j,i] = gsw.density.rho(SA_loc_jun[t,k,j,i], CT_loc_jun[t,k,j,i], pressure_loc[k])  

    
print("Writing the file for June")

bdy_file = nc.Dataset(path_to_save + 'NEP36_T_S_Spice_june.nc', 'w', zlib=True);

bdy_file.createDimension('x', sal_june.shape[3]);
bdy_file.createDimension('y', sal_june.shape[2]);
bdy_file.createDimension('deptht', sal_june.shape[1]);
bdy_file.createDimension('time_counter', None);

x = bdy_file.createVariable('x', 'int32', ('x',), zlib=True);
x.units = 'indices';
x.longname = 'x indices of NEP36';

y = bdy_file.createVariable('y', 'int32', ('y',), zlib=True);
y.units = 'indices';
y.longname = 'y indices of NEP36';

deptht = bdy_file.createVariable('deptht', 'float32', ('deptht',), zlib=True);
deptht.units = 'm';
deptht.longname = 'Vertical T Levels';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';


votemper  = bdy_file.createVariable('votemper', 'float32', ('time_counter', 'deptht', 'y', 'x'), zlib=True);
vosaline  = bdy_file.createVariable('vosaline', 'float32', ('time_counter', 'deptht', 'y', 'x'), zlib=True);
spiciness = bdy_file.createVariable('spiciness', 'float32', ('time_counter', 'deptht', 'y', 'x'), zlib=True);
density   = bdy_file.createVariable('density', 'float32', ('time_counter', 'deptht', 'y', 'x'), zlib=True) 

votemper[...]  = temp_june[...];
vosaline[...]  = sal_june[...];
spiciness[...] = spic_jun[...];
density[...]   = rho_jun[...];

bdy_file.close()
    
print("The June file is successfully written")  


print("End of Script: Thanks")



    
