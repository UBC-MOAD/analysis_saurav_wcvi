import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import cmocean as cmo
import matplotlib.gridspec as gridspec
from scipy.io import loadmat
from scipy.interpolate import griddata
from dateutil.parser import parse
from salishsea_tools import geo_tools, viz_tools, tidetools, nc_tools
import gsw

import seaborn as sns

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht'][:32]
y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))


#nc_file = nc.Dataset('/data/ssahu/ARIANE/LB_08/LB_08_big_sequential.nc')

#nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/isopyncal_263/right_traj_all_down.nc')

#nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/isopyncal_263/right_traj_all_down_high_spice.nc')

#nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/accurate_isopyncal_particle_positions/eddy_water.nc')

#nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/accurate_isopyncal_particle_positions/south_outer_shelf_water.nc')

#nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/accurate_isopycnals_eddy_water_particles_two_months_more.nc')

nc_file = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/\
accurate_isopycnals_south_outer_shelf__water_particles_two_months_more.nc')

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

x2=nc_file.variables['final_x'][:]
y2=nc_file.variables['final_y'][:]


final_z = nc_file.variables['final_z'][:]

final_age_days = final_age[:]/(3600)

x_final = []
y_final = []


for i in np.arange(x2.shape[0]):
    x_final = np.append(arr=x_final,values=x_wcvi_slice[np.int(np.rint(x2[i]))-1])
    y_final = np.append(arr=y_final,values=y_wcvi_slice[np.int(np.rint(y2[i]))-1])



bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')

Z = bathy.variables['Bathymetry']


lon = bathy['nav_lon'][...]
lat = bathy['nav_lat'][...]




cmap=plt.cm.get_cmap('nipy_spectral')

cmap.set_bad('#8b7765')
cmin = 0
cmax = 300

import matplotlib as mpl


fig, ax = plt.subplots(1, 1, figsize=(16,12)); ax.grid()
CS = ax.contour(x_wcvi_slice,y_wcvi_slice,Z[y_wcvi_slice,x_wcvi_slice], np.arange(200,210,10))



ax.set_xlabel('x index', fontsize =16)
ax.set_ylabel('y index', fontsize = 16)
ax.tick_params(axis='both',labelsize =16)


ax.legend(loc = 'best')

viz_tools.plot_land_mask(ax, bathy, yslice=y_wcvi_slice, xslice=x_wcvi_slice, color='burlywood')
viz_tools.plot_coastline(ax, bathy, yslice=y_wcvi_slice, xslice=x_wcvi_slice, color='brown')

x = [510,575]
y = [320,200]
ax.plot(x, y , 'bo-')

xmin = x_wcvi_slice[np.int(np.min(np.unique(x1)))]-1
xmax = x_wcvi_slice[np.int(np.max(np.unique(x1)))]-1


ymin = y_wcvi_slice[np.int(np.min(np.unique(y1)))]-1
ymax  = y_wcvi_slice[np.int(np.max(np.unique(y1)))]-1


x_1 = [xmin, xmax]
y_1 = [ymin, ymin]
ax.plot(x_1, y_1 , 'go-')

x_2 = [xmin, xmax]
y_2 = [ymax, ymax]
ax.plot(x_2, y_2 , 'go-')

x_3 = [xmin, xmin]
y_3 = [ymin, ymax]
ax.plot(x_3, y_3 , 'go-')

x_4 = [xmax, xmax]
y_4 = [ymin, ymax]
ax.plot(x_4, y_4 , 'go-')

ax.vlines(x = 635, ymin = 261, ymax = 279, color = 'blue')

lon_LB08 = -125.4775
lat_LB08 = 48.4217

j, i = geo_tools.find_closest_model_point(lon_LB08,lat_LB08,\
                                          lon,lat,grid='NEMO',tols=\
                                          {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                           'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}}) 

ax.scatter(i,j, marker = 'o', c = 'red', s = 300, linewidths=100, label = 'Eddy region')

ax.hlines(y = 200, xmin = 575, xmax=645, color = 'b')
ax.hlines(y = 320, xmin = 510, xmax=590, color = 'b')
ax.grid()

contour = CS.collections[0]
vs = contour.get_paths()[0].vertices


vert = CS.collections[0].get_paths()[0].vertices

x_200_wcvi = vert[:,0]
y_200_wcvi = vert[:,1]



Line = ax.plot(x_200_wcvi[::10], y_200_wcvi[::10], 'go-')
fig.tight_layout()

plt.close()

#cross_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_savedcross_index.npy')
#time_index  = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/time_index.npy')

#cross_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/cross_index_all_down.npy')

#time_index  = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/time_index_all_down.npy')

#cross_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/cross_index_all_down_high_spice.npy')

#time_index  = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/time_index_all_down_high_spice.npy')

#cross_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/cross_index_accurate_iso_low.npy')

#time_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/time_index_accurate_iso_low.npy')

#cross_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/cross_index_accurate_iso_high.npy')

#time_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/time_index_accurate_iso_high.npy')


#cross_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/cross_index_accurate_iso_low_two_more_months.npy')

#time_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/time_index_accurate_iso_low_two_more_months.npy')

cross_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/cross_index_accurate_iso_high_two_more_months.npy')

time_index = np.load(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/time_index_accurate_iso_high_two_more_months.npy')

y_200_loc = np.empty_like(cross_index)
x_200_loc = np.empty_like(cross_index)

for m in np.arange(cross_index.shape[0]):
    
    for k in np.arange(1,traj_lon.shape[0]-1):


        y, x = geo_tools.find_closest_model_point(traj_lon[k,cross_index[m].astype(int)],\
                                              traj_lat[k,cross_index[m].astype(int)],\
                                  lon,lat,grid='NEMO',tols=\
                                  {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                   'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
        y1,x1 = geo_tools.find_closest_model_point(traj_lon[k+1,cross_index[m].astype(int)],\
                                                   traj_lat[k+1,cross_index[m].astype(int)],\
                                      lon,lat,grid='NEMO',tols=\
                                      {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                       'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})

        bbox = mpl.transforms.Bbox(points=[(x,y), (x1,y1)]) 

        PATH = mpl.path.Path(vertices =[(x,y), (x1,y1)] )

        #         if (mpl.path.Path(vertices=vert).intersects_bbox(bbox, filled=True)) == True:

        if (mpl.path.Path(vertices=vert).intersects_path(PATH, filled=True)) == True:

            y_200_loc[m], x_200_loc[m] = y,x
            
            print(m)

            break
            
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_200_total.npy', arr=x_200_loc)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_200_total.npy', arr=y_200_loc)

#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/x_200_total_all_down_high_spice.npy', arr=x_200_loc)
#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/y_200_total_all_down_high_spice.npy', arr=y_200_loc)


#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/x_200_total_eddy_water_acc_iso.npy', arr=x_200_loc)

#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/y_200_total_eddy_water_acc_iso.npy', arr=y_200_loc)

#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/x_200_south_outer_shelf_water_acc_iso.npy', arr=x_200_loc)

#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/y_200_south_outer_shelf_water_acc_iso.npy', arr=y_200_loc)

#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/x_200_total_eddy_water_acc_iso_two_more_months.npy', arr=x_200_loc)

#np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/y_200_total_eddy_water_acc_iso_iso_two_more_months.npy', arr=y_200_loc)

np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/x_200_south_outer_shelf_water_acc_iso_two_more_months.npy', arr=x_200_loc)

np.save(file='/data/ssahu/NEP36_2013_summer_hindcast/Ariane_files/numpy_arrays_saved/y_200_south_outer_shelf_water_acc_iso_two_more_months.npy', arr=y_200_loc)
