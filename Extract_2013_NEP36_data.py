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

zlevels = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d_20130429_20131025_grid_T_20130429-20130508.nc').variables['deptht']

lon = bathy['nav_lon'][...]
lat = bathy['nav_lat'][...]

z0 = np.ma.masked_values(Z, 0)

y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))

def tem_sal_timeseries_at_WCVI_locations(grid_scalar):#, j, i):

    temp = grid_scalar.variables['temp'][:, :, 1:, 1:]
    sal =  grid_scalar.variables['salt'][:, :, 1:, 1:]
    
    scalar_ts = namedtuple('scalar_ts', 'temp, sal')

    return scalar_ts(temp, sal)

temp = np.empty((180,32,y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
sal = np.empty((180,32,y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))


i = 0
for file in sorted(glob.glob('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d_*_grid_T_*.nc')):

    scalar_ts = tem_sal_timeseries_at_WCVI_locations(nc.Dataset(file))
    temp[i:i+10,...] = scalar_ts[0]
    sal[i:i+10,...] = scalar_ts[1]
    i = i+10

T_2013_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d_20130429_20131025_grid_T_20130429-20130508.nc')
    
SA_loc = np.empty_like(sal)
CT_loc = np.empty_like(sal)
spic = np.empty_like(sal)
rho = np.empty_like(sal)

lat = T_2013_file.variables['nav_lat'][1:,1:]
lon = T_2013_file.variables['nav_lon'][1:,1:]

pressure_loc = gsw.p_from_z(-zlevels[:],np.mean(lat))

for t in np.arange(sal.shape[0]):
    for k in np.arange(sal.shape[1]):
        for j in np.arange(sal.shape[2]):
            for i in np.arange(sal.shape[3]):
                SA_loc[t,k,j,i] = gsw.SA_from_SP(sal[t,k,j,i], pressure_loc[k], lon[j,i], lat[j,i])
                CT_loc[t,k,j,i] = gsw.CT_from_pt(sal[t,k,j,i], temp[t,k,j,i])
                spic[t,k,j,i] = gsw.spiciness0(SA_loc[t,k,j,i],CT_loc[t,k,j,i])
#                rho_jun[t,k,j,i] = gsw.density.rho(SA_loc_jun[t,k,j,i], CT_loc_jun[t,k,j,i], pressure_loc[k])  
                rho[t,k,j,i] = gsw.density.rho(SA_loc[t,k,j,i], CT_loc[t,k,j,i], 0)

    
print("Beginning to write the output file")

bdy_file = nc.Dataset(path_to_save + 'NEP36_2013_T_S_Spice_larger_offshore_rho_correct.nc', 'w', zlib=True);

bdy_file.createDimension('x', sal.shape[3]);
bdy_file.createDimension('y', sal.shape[2]);
bdy_file.createDimension('deptht', sal.shape[1]);
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

votemper[...]  = temp[...];
vosaline[...]  = sal[...];
spiciness[...] = spic[...];
density[...]   = rho[...];

bdy_file.close()
    
print("The file is successfully written")  


print("End of Script: Thanks")






