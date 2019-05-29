import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.ma as ma
import glob
from collections import namedtuple, OrderedDict
import netCDF4 as nc
import os
import scipy
import scipy.io as sio
from scipy import interpolate, signal
from pyproj import Proj,transform
import sys
sys.path.append('/ocean/ssahu/CANYONS/wcvi/grid/')
from bathy_common import *
from matplotlib import path
from salishsea_tools import viz_tools
import xarray as xr
from salishsea_tools import nc_tools
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import cmocean as cmo
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from scipy.interpolate import griddata
from dateutil.parser import parse
from salishsea_tools import geo_tools, viz_tools, tidetools, nc_tools
import gsw
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Ellipse
import matplotlib as mpl

import seaborn as sns
from windrose import plot_windrose
from windrose import WindroseAxes


from dateutil        import parser
from datetime import datetime

import numpy.ma as ma

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht'][:32]
y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))


bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')

Z = bathy.variables['Bathymetry']


lon = bathy['nav_lon'][...]
lat = bathy['nav_lat'][...]



#nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/accurate_isopyncal_particle_positions/\
#eddy_water.nc')


nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/accurate_isopyncal_particle_positions/\
south_outer_shelf_water.nc')

init_x = nc_file.variables['init_x']
init_z = nc_file.variables['init_z']
init_age = nc_file.variables['init_age']

traj_depth = nc_file.variables['traj_depth'][:]
traj_lon   = nc_file.variables['traj_lon'][:]
traj_lat   = nc_file.variables['traj_lat'][:]
traj_rho   = nc_file.variables['traj_dens'][:]
traj_tem   = nc_file.variables['traj_temp'][:]
traj_sal   = nc_file.variables['traj_salt'][:]
traj_time  = nc_file.variables['traj_time']


final_age = nc_file.variables['final_age']


# lon1=nc_file.variables['traj_lon'][:]
# lat1=nc_file.variables['traj_lat'][:]
# dep1=nc_file.variables['traj_depth'][:]
x1=nc_file.variables['init_x'][:]
y1=nc_file.variables['init_y'][:]
t1=nc_file.variables['traj_time'][:]

x_low = x1
y_low = y1

x2=nc_file.variables['final_x'][:]
y2=nc_file.variables['final_y'][:]


final_z = nc_file.variables['final_z'][:]

final_age_days = final_age[:]/(3600)


x_final = []
y_final = []


for i in np.arange(x2.shape[0]):
    x_final = np.append(arr=x_final,values=x_wcvi_slice[np.int(np.rint(x2[i]))-1])
    y_final = np.append(arr=y_final,values=y_wcvi_slice[np.int(np.rint(y2[i]))-1])
    
x = [510,575]
y = [320,200]


coefficients = np.polyfit(x, y, 1)

x_final_off = np.arange(x[0],x[1])
polynomial = np.poly1d(coefficients)
y_final_off = np.rint(polynomial(x_final_off))

p1= (x[0],y[0])
p1 = np.asarray(p1)
p2 = (x[1], y[1])
p2 = np.asarray(p2)



d = np.empty_like(x_final)

off_ind = []


for r in np.arange(x_final.shape[0]):

    p3 = (x_final[r], y_final[r])
    p3 = np.asarray(p3)
    d[r] =np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)
    
    if ((d[r] < 0) & (x_final[r] < 575)):
        off_ind = np.append(arr=off_ind, values=np.int(r))
        
        
num_north = np.round(a=final_age[(x_final < 580) & (y_final > 300)].shape[0]/final_z.shape[0], decimals=3)

north_percent = 100*num_north

print(north_percent)

num_cuc = np.round(a=np.where(final_z[((x_final < 630) & (y_final < 200)) & \
                                      ((x_final > 580) & (y_final < 200))]>26.3)[0].shape[0]/final_z.shape[0], decimals = 3)

cuc_percent = 100*num_cuc

print(cuc_percent)

num_south = np.round(a=np.where(final_z[((x_final < 630) & (y_final < 200)) & \
                                      ((x_final > 580) & (y_final < 200))]<=26.3)[0].shape[0]/final_z.shape[0], decimals = 3)

south_percent = 100*num_south

print(south_percent)

### Bigger box
x = [510,575]
y = [320,200]


coefficients = np.polyfit(x, y, 1)

x_final_off = np.arange(x[0],x[1])
polynomial = np.poly1d(coefficients)
y_final_off = np.rint(polynomial(x_final_off))

num_off = off_ind.shape[0]/final_z.shape[0]
off_percent = np.round(a=100*num_off, decimals=1)

print(off_percent)

num_juan = np.round(a=np.where((x_final > 648))[0].shape[0]/final_z.shape[0], decimals=3)

juan_percent = num_juan*100

print(juan_percent)

cuc_percent_low = cuc_percent
juan_percent_low = juan_percent
north_percent_low = north_percent
south_percent_low = south_percent
off_percent_low = off_percent


traj_tem_last   = np.empty_like(final_age_days)
traj_sal_last   = np.empty_like(final_age_days)
traj_rho_last   = np.empty_like(final_age_days)
traj_depth_last = np.empty_like(final_age_days)


traj_tem_init   = np.empty_like(final_age_days)
traj_sal_init   = np.empty_like(final_age_days)
traj_rho_init   = np.empty_like(final_age_days)
traj_depth_init = np.empty_like(final_age_days)

for r in np.arange(traj_tem_last.shape[0]):
    traj_tem_last[r]  =  traj_tem[np.int(final_age_days[r]),r]
    traj_sal_last[r]  =  traj_sal[np.int(final_age_days[r]),r]
    traj_rho_last[r]  =  traj_rho[np.int(final_age_days[r]),r]
    traj_depth_last[r] = traj_depth[np.int(final_age_days[r]),r]
    
    traj_tem_init[r]  =  traj_tem[0,r]
    traj_sal_init[r]  =  traj_sal[0,r]
    traj_rho_init[r]  =  traj_rho[0,r]    
    traj_depth_init[r]=  traj_depth[0,r] 
    
    
max_life_ind = np.array(np.where(final_age_days == np.max(final_age_days))[0])

off_ind   = off_ind
north_ind = np.where([(x_final < 580) & (y_final > 300)])[1]
south_ind = []
cuc_ind   = []


ind_all_south  = np.where(((x_final < 630) & (y_final < 200)) & \
                                      ((x_final > 580) & (y_final < 200)))[0]

for k in ind_all_south:
    if final_z[k] <= 26.3:
        south_ind = np.append(arr=south_ind, values= k)
        
    if final_z[k] > 26.3:
        cuc_ind = np.append(arr=cuc_ind, values=k)
        

juan_ind =  np.where((x_final > 648))[0]

non_local_ind = np.concatenate((off_ind, north_ind, south_ind, cuc_ind, juan_ind))

ind_local = np.setdiff1d(max_life_ind, np.unique(non_local_ind, return_index= False).astype(int), assume_unique=True)

# off_ind/

# ind_local = np.setdiff1d(max_life_ind, off_ind, assume_unique= True)

x_local_low_actual = x2[ind_local]
y_local_low_actual = y2[ind_local]

depth_local_low_actual = -traj_depth_last[ind_local]

bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')

Z = bathy.variables['Bathymetry']


lon = bathy['nav_lon'][...]
lat = bathy['nav_lat'][...]



file_model = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d_20130429_20131025_grid_T_20130429-20130508.nc')

lon_small = file_model.variables['nav_lon'][1:,1:]
lat_small = file_model.variables['nav_lat'][1:,1:]

y_plot_cuc = []
x_plot_cuc = []

for part in cuc_ind[:].astype(int):
    
    print(part)

    for k in np.arange(np.int(final_age_days[part])):
        
        y, x = geo_tools.find_closest_model_point(traj_lon[k,part],traj_lat[k,part],\
                                      lon,lat,grid='NEMO',tols=\
                                      {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                       'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
        y_plot_cuc = np.append(arr=y_plot_cuc, values=y)
        x_plot_cuc = np.append(arr=x_plot_cuc, values=x)


#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_cuc_eddy_water_points.npy', arr=y_plot_cuc)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_cuc_eddy_water_points.npy', arr=x_plot_cuc)

np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_cuc_south_outer_points.npy', arr=y_plot_cuc)
np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_cuc_south_outer_points.npy', arr=x_plot_cuc)

print("Thanks the script has run to completion for CUC")


y_plot_south = []
x_plot_south = []

for part in south_ind[:].astype(int):
    
    print(part)

    for k in np.arange(np.int(final_age_days[part])):
        
        y, x = geo_tools.find_closest_model_point(traj_lon[k,part],traj_lat[k,part],\
                                      lon,lat,grid='NEMO',tols=\
                                      {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                       'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
        y_plot_south = np.append(arr=y_plot_south, values=y)
        x_plot_south = np.append(arr=x_plot_south, values=x)


#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_south_eddy_water_points.npy', arr=y_plot_south)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_south_eddy_water_points.npy', arr=x_plot_south)

np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_south_south_outer_points.npy', arr=y_plot_south)
np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_south_south_outer_points.npy', arr=x_plot_south)

print("Thanks the script has run to completion for South")

y_plot_off = []
x_plot_off = []

for part in off_ind[:].astype(int):
    
    print(part)

    for k in np.arange(np.int(final_age_days[part])):
        
        y, x = geo_tools.find_closest_model_point(traj_lon[k,part],traj_lat[k,part],\
                                      lon,lat,grid='NEMO',tols=\
                                      {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                       'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
        y_plot_off = np.append(arr=y_plot_off, values=y)
        x_plot_off = np.append(arr=x_plot_off, values=x)


#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_off_eddy_water_points.npy', arr=y_plot_off)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_off_eddy_water_points.npy', arr=x_plot_off)

np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_off_south_outer_points.npy', arr=y_plot_off)
np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_off_south_outer_points.npy', arr=x_plot_off)

print("Thanks the script has run to completion for Off")

y_plot_local = []
x_plot_local = []

for part in ind_local[:].astype(int):
    
    print(part)

    for k in np.arange(np.int(final_age_days[part])):
        
        y, x = geo_tools.find_closest_model_point(traj_lon[k,part],traj_lat[k,part],\
                                      lon,lat,grid='NEMO',tols=\
                                      {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                       'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
        y_plot_local = np.append(arr=y_plot_local, values=y)
        x_plot_local = np.append(arr=x_plot_local, values=x)


#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_local_eddy_water_points.npy', arr=y_plot_local)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_local_eddy_water_points.npy', arr=x_plot_local)


np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_local_south_outer_points.npy', arr=y_plot_local)
np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_local_south_outer_points.npy', arr=x_plot_local)

print("Thanks the script has run to completion for Local")

north_ind = north_ind[np.where(x_final[north_ind].astype(int) > 500)]

y_plot_north = []
x_plot_north = []

for part in north_ind[:].astype(int):
    
    print(part)

    for k in np.arange(np.int(final_age_days[part])):
        
        y, x = geo_tools.find_closest_model_point(traj_lon[k,part],traj_lat[k,part],\
                                      lon,lat,grid='NEMO',tols=\
                                      {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                       'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
        y_plot_north = np.append(arr=y_plot_north, values=y)
        x_plot_north = np.append(arr=x_plot_north, values=x)


#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_north_eddy_water_points.npy', arr=y_plot_north)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_north_eddy_water_points.npy', arr=x_plot_north)

np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_north_south_outer_points.npy', arr=y_plot_north)
np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_north_south_outer_points.npy', arr=x_plot_north)


print("Thanks the script has run to completion for North")

y_plot_juan = []
x_plot_juan = []

for part in juan_ind[:].astype(int):
    
    print(part)

    for k in np.arange(np.int(final_age_days[part])):
        
        y, x = geo_tools.find_closest_model_point(traj_lon[k,part],traj_lat[k,part],\
                                      lon,lat,grid='NEMO',tols=\
                                      {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                       'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
        y_plot_juan = np.append(arr=y_plot_juan, values=y)
        x_plot_juan = np.append(arr=x_plot_juan, values=x)


#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_juan_eddy_water_points.npy', arr=y_plot_juan)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_juan_eddy_water_points.npy', arr=x_plot_juan)

np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_juan_south_outer_points.npy', arr=y_plot_juan)
np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_juan_south_outer_points.npy', arr=x_plot_juan)


print("Thanks the script has run to completion for Juan")


print("Thanks, the entire script has run to completion")



