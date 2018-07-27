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
from grid_alignment import calculate_initial_compass_bearing as cibc
from bathy_common import *
from matplotlib import path
import xarray as xr
import pandas as pd
import scipy.io as sio
import matplotlib.cm as cm
import cmocean as cmo
import matplotlib.gridspec as gridspec
from dateutil.parser import parse
from salishsea_tools import geo_tools, viz_tools, tidetools, nc_tools
import gsw
from scipy.interpolate import interp1d
import os

path_to_save ='/data/ssahu/NEP36_Extracted_Months/' #'/home/ssahu/saurav/'


bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')

Z = bathy.variables['Bathymetry'][:]

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht'][:32]


mask = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/mesh_mask.nc')

tmask = mask.variables['tmask'][0,:32,180:350, 480:650]
umask = mask.variables['umask'][0,:32,180:350, 480:650]
vmask = mask.variables['vmask'][0,:32,180:350, 480:650]
mbathy = mask.variables['mbathy'][0,180:350, 480:650]


y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))



mbathy[mbathy>32] = 32

NEP_2013 = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/NEP36_2013_T_S_Spice_larger_offshore_rho_correct.nc')

rho = NEP_2013.variables['density']


def U_timeseries_at_WCVI_locations(grid_U):
    
    u_vel = grid_U.variables['uo'][:,:,:,:]

    
    vector_u = namedtuple('vector_u', 'u_vel')

    return vector_u(u_vel)


def V_timeseries_at_WCVI_locations(grid_V):
    
    v_vel = grid_V.variables['vo'][:,:,:,:]

    
    vector_v = namedtuple('vector_v', 'v_vel')

    return vector_v(v_vel)



u_vel = np.empty((180,zlevels.shape[0],1+y_wcvi_slice.shape[0],1+x_wcvi_slice.shape[0]))
v_vel = np.empty((180,zlevels.shape[0],1+y_wcvi_slice.shape[0],1+x_wcvi_slice.shape[0]))



i = 0

for file in sorted(glob.glob('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d*grid_U*.nc')):
    vector_u = U_timeseries_at_WCVI_locations(nc.Dataset(file))
    u_vel[i:i+10,...] = vector_u[0]
    i = i+10

j = 0
for file in sorted(glob.glob('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d*grid_V*.nc')):
    vector_v = V_timeseries_at_WCVI_locations(nc.Dataset(file))
    v_vel[j:j+10,...] = vector_v[0]
    j = j+10
  

  

u_tzyx = np.empty((u_vel.shape[0],zlevels.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
v_tzyx = np.empty_like(u_tzyx)

for t in np.arange(u_tzyx.shape[0]):
    for level in np.arange(zlevels.shape[0]):
        u_tzyx[t, level,...], v_tzyx[t, level,...] = viz_tools.unstagger(u_vel[t,level,...], v_vel[t, level,...])
        u_tzyx[t, level,...] = np.ma.masked_array(u_tzyx[t, level,...], mask= 1- umask[level,:,:,])
        v_tzyx[t, level,...] = np.ma.masked_array(v_tzyx[t, level,...], mask= 1- vmask[level,:,:])



znew = np.arange(0,250,0.1)
den = np.arange(26,26.5,0.1)
tol = 0.01




print("Starting interpolation and data extraction")

u_vel_time_iso = np.empty((u_tzyx.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
v_vel_time_iso = np.empty((v_tzyx.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))


for t in np.arange(u_vel_time_iso.shape[0]):

    rho_0  = rho[t, :, :, :] - 1000
    u_0    = u_tzyx[t, :, :, :]
    v_0    = v_tzyx[t,:,:,:]


    u_spec_iso = np.empty((den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
    v_spec_iso = np.empty((den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))


    for iso in np.arange(den.shape[0]):

        u_den = np.empty((y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
        v_den = np.empty((y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))


        for j in np.arange(y_wcvi_slice.shape[0]):


            u_iso = np.empty(x_wcvi_slice.shape[0])
            v_iso = np.empty(x_wcvi_slice.shape[0])



            rho_new  = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
            u_new    = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
            v_new    = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))


            for i in np.arange(rho_new.shape[1]):


                f = interp1d(zlevels[:],rho_0[:,j,i],fill_value='extrapolate')
                g = interp1d(zlevels[:],u_0[:,j,i],fill_value='extrapolate')
                h = interp1d(zlevels[:],v_0[:,j,i],fill_value='extrapolate')



                rho_new[:,i]  = f(znew[:])
                u_new[:,i]    = g(znew[:])
                v_new[:,i]    = h(znew[:])


                V = rho_new[:,i]

                ind = (V>den[iso]-tol)&(V<den[iso]+tol)

                u_iso[i] = np.nanmean(u_new[ind,i])
                v_iso[i] = np.nanmean(v_new[ind,i])


                u_den[j,i] = u_iso[i]
                v_den[j,i] = v_iso[i]

                
                u_spec_iso[iso,j,i] = u_den[j,i]
                v_spec_iso[iso,j,i] = v_den[j,i]


                u_vel_time_iso[t,iso,j,i] = u_spec_iso[iso,j,i]
                v_vel_time_iso[t,iso,j,i] = v_spec_iso[iso,j,i]




print("Writing the isopycnal data")


bdy_file = nc.Dataset(path_to_save + 'short_NEP36_2013_along_isopycnal_larger_offshore_velocities.nc', 'w', zlib=True);

bdy_file.createDimension('x', u_vel_time_iso.shape[3]);
bdy_file.createDimension('y', u_vel_time_iso.shape[2]);
bdy_file.createDimension('isot', u_vel_time_iso.shape[1]);
bdy_file.createDimension('time_counter', None);


x = bdy_file.createVariable('x', 'int32', ('x',), zlib=True);
x.units = 'indices';
x.longname = 'x indices of NEP36';

y = bdy_file.createVariable('y', 'int32', ('y',), zlib=True);
y.units = 'indices';
y.longname = 'y indices of NEP36';

isot = bdy_file.createVariable('isot', 'float32', ('isot',), zlib=True);
isot.units = 'm';
isot.longname = 'Vertical isopycnal Levels';

time_counter = bdy_file.createVariable('time_counter', 'int32', ('time_counter',), zlib=True);
time_counter.units = 's';
time_counter.longname = 'time';


u_velocity = bdy_file.createVariable('u_velocity', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)
v_velocity = bdy_file.createVariable('v_velocity', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)







u_velocity[...] = u_vel_time_iso[...];
v_velocity[...] = u_vel_time_iso[...];

isot[...] = den[:];
x[...] = x_wcvi_slice[:];
y[...] = y_wcvi_slice[:];

bdy_file.close()

                                                                                                                                                                                        

