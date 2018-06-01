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

path_to_save ='/data/ssahu/NEP36_Extracted_Months/' #'/home/ssahu/saurav/'


bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')

Z = bathy.variables['Bathymetry'][:]

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht']

lon = bathy['nav_lon'][...]
lat = bathy['nav_lat'][...]

z0 = np.ma.masked_values(Z, 0)

y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))

def tem_sal_timeseries_at_WCVI_locations(grid_scalar):#, j, i):

    temp = grid_scalar.variables['votemper'][0,:, :, :]
    sal = grid_scalar.variables['vosaline'][0,:, :, :]
    
    scalar_ts = namedtuple('scalar_ts', 'temp, sal')

    return scalar_ts(temp, sal)

print("Extracting August Data")

temp_aug = np.empty((31,50,Z.shape[0],Z.shape[1]))
sal_aug = np.empty((31,50,Z.shape[0],Z.shape[1]))

i = 0
for file in sorted(glob.glob('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_201508*grid_T.nc')):
    
    scalar_ts = tem_sal_timeseries_at_WCVI_locations(nc.Dataset(file))
    temp_aug[i,...] = scalar_ts[0]
    sal_aug[i,...] = scalar_ts[1]
    i = i+1

print("Extracting July Data")    
    
    
temp_july = np.empty((31,50,Z.shape[0],Z.shape[1]))
sal_july = np.empty((31,50,Z.shape[0],Z.shape[1]))


i = 0
for file in sorted(glob.glob('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_201507*grid_T.nc')):

    scalar_ts = tem_sal_timeseries_at_WCVI_locations(nc.Dataset(file))
    temp_july[i,...] = scalar_ts[0]
    sal_july[i,...] = scalar_ts[1]
    i = i+1

print("Calculating the Spice for July for the WCVI subset; the rest of the locations will have empty spice")
    
pressure_loc = gsw.p_from_z(-zlevels[:],np.mean(lat))

SA_loc_jul = np.empty_like(sal_july)
CT_loc_jul = np.empty_like(sal_july)
spic_jul = np.empty_like(sal_july)
rho_jul = np.empty_like(sal_july)

for t in np.arange(sal_july.shape[0]):
    for k in np.arange(sal_july.shape[1]):
        for j in np.arange(180,350):
            for i in np.arange(480,650):
                SA_loc_jul[t,k,j,i] = gsw.SA_from_SP(sal_july[t,k,j,i], pressure_loc[k], lon[j,i], lat[j,i])
                CT_loc_jul[t,k,j,i] = gsw.CT_from_pt(sal_july[t,k,j,i], temp_july[t,k,j,i])
                spic_jul[t,k,j,i] = gsw.spiciness0(SA_loc_jul[t,k,j,i],CT_loc_jul[t,k,j,i])
                rho_jul[t,k,j,i] = gsw.density.rho(SA_loc_jul[t,k,j,i], CT_loc_jul[t,k,j,i], pressure_loc[k])  
print("Calculating the Spice for August for the WCVI subset; the rest of the locations will have empty spice")
    
    
SA_loc_aug = np.empty_like(sal_aug)
CT_loc_aug = np.empty_like(sal_aug)
spic_aug = np.empty_like(sal_aug)
rho_aug = np.empty_like(sal_aug)

for t in np.arange(sal_aug.shape[0]):
    for k in np.arange(sal_aug.shape[1]):
        for j in np.arange(180,350):
            for i in np.arange(480,650):
                SA_loc_aug[t,k,j,i] = gsw.SA_from_SP(sal_aug[t,k,j,i], pressure_loc[k], lon[j,i], lat[j,i])
                CT_loc_aug[t,k,j,i] = gsw.CT_from_pt(sal_aug[t,k,j,i], temp_aug[t,k,j,i])
                spic_aug[t,k,j,i] = gsw.spiciness0(SA_loc_aug[t,k,j,i],CT_loc_aug[t,k,j,i])
                rho_aug[t,k,j,i] = gsw.density.rho(SA_loc_aug[t,k,j,i], CT_loc_aug[t,k,j,i], pressure_loc[k])    
    
print("Writing the file for July")

bdy_file = nc.Dataset(path_to_save + 'NEP36_T_S_Spice_july_larger_offshore.nc', 'w', zlib=True);

bdy_file.createDimension('x', sal_july.shape[3]);
bdy_file.createDimension('y', sal_july.shape[2]);
bdy_file.createDimension('deptht', sal_july.shape[1]);
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

votemper[...]  = temp_july[...];
vosaline[...]  = sal_july[...];
spiciness[...] = spic_jul[...];
density[...]   = rho_jul[...];

bdy_file.close()
    
print("The July file is successfully written")  


print("Writing the file for August")

bdy_file = nc.Dataset(path_to_save + 'NEP36_T_S_Spice_aug_larger_offshore.nc', 'w', zlib=True);

bdy_file.createDimension('x', sal_aug.shape[3]);
bdy_file.createDimension('y', sal_aug.shape[2]);
bdy_file.createDimension('deptht', sal_aug.shape[1]);
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
spiciness = bdy_file.createVariable('spiciness', 'float32', ('time_counter', 'deptht', 'y', 'x'), zlib=True)
density   = bdy_file.createVariable('density', 'float32', ('time_counter', 'deptht', 'y', 'x'), zlib=True)


votemper[...]  = temp_aug[...];
vosaline[...]  = sal_aug[...];
spiciness[...] = spic_aug[...];
density[...]   = rho_aug[...];

bdy_file.close()
    
print("The August file is successfully written") 


print("End of Script: Thanks")



    
