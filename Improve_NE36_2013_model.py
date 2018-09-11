from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import os
import glob
import fnmatch
from collections import namedtuple, OrderedDict
import scipy.io as sio
from scipy.interpolate import interp1d
import matplotlib.cm as cm
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
import pandas as pd
import seaborn as sns
from seabird.cnv import fCNV
import gsw
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit


iso_NEP = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/2013_short_slice_NEP36_along_isopycnal_larger_offshore_rho_correct.nc')

iso_spic = iso_NEP.variables['spiciness']
isot = iso_NEP.variables['isot']

vel_iso_NEP = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/short_NEP36_2013_along_isopycnal_larger_offshore_velocities.nc')
u_vel_iso = vel_iso_NEP.variables['u_velocity']
v_vel_iso = vel_iso_NEP.variables['v_velocity']
isot_vel = vel_iso_NEP.variables['isot']

bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')


Z = bathy.variables['Bathymetry']

y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))


lon = bathy['nav_lon'][180:350, 480:650]
lat = bathy['nav_lat'][180:350, 480:650]

lon_model = bathy['nav_lon'][...]
lat_model = bathy['nav_lat'][...]


NEP = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/NEP36_2013_T_S_Spice_larger_offshore_rho_correct.nc')


sal = NEP.variables['vosaline']
temp = NEP.variables['votemper']
spic = NEP.variables['spiciness']
rho = NEP.variables['density']

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht'][:32]


mask = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/mesh_mask.nc')

tmask = mask.variables['tmask'][0,:32,180:350, 480:650]
umask = mask.variables['umask'][0,:32,180:350, 480:650]
vmask = mask.variables['vmask'][0,:32,180:350, 480:650]
mbathy = mask.variables['mbathy'][0,180:350, 480:650]

def U_timeseries_at_WCVI_locations(grid_U):
    
    u_vel = grid_U.variables['uo'][:,:,1:,1:]
   
    vector_u = namedtuple('vector_u', 'u_vel')

    return vector_u(u_vel)


def V_timeseries_at_WCVI_locations(grid_V):
    
    v_vel = grid_V.variables['vo'][:,:,1:,1:]
    
    vector_v = namedtuple('vector_v', 'v_vel')

    return vector_v(v_vel)



u_vel = np.empty((180,zlevels.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
v_vel = np.empty((180,zlevels.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))



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
    
file = []

i = 0
for file_number in sorted(glob.glob('/home/ssahu/saurav/Falkor_code/*.cnv')):
    
    file.append(file_number)
    
    i = i+1

temp_location = []
sal_location = []
lat_location = []
lon_location = []
z_location = []
time_location = []

for example in file:
    time_location = np.append(arr= time_location, values= example[51:-4])
    
time_location = time_location.astype(int)
time_location[:] = time_location[:] + 93 

def falkor_locations(profile):

    pressure = profile['PRES'][:]
    PT = profile['potemperature'][:]
    T = profile['TEMP'][:]
    SP = profile['PSAL'][:]
    lat = np.nanmean(profile['LATITUDE'][:])
    lon = np.nanmean(profile['LONGITUDE'][:])
    
    z = gsw.z_from_p(-pressure, lat)
    
    falkor_scalar_ts = namedtuple('falkor_scalar_ts', 'temp, sal, lat, lon, z')

    return falkor_scalar_ts(PT, SP, lat, lon, z)

for j in np.arange(len(file)):

    falkor_scalar_ts = falkor_locations(fCNV(file[j]))
    
    
    temp_that_file = np.empty(falkor_scalar_ts[0].shape)
    temp_that_file = falkor_scalar_ts[0]
    temp_location.append(temp_that_file)
    
    sal_that_file = np.empty(falkor_scalar_ts[1].shape)
    sal_that_file = falkor_scalar_ts[1]
    sal_location.append(sal_that_file)
    
    
    z_that_file = np.empty(falkor_scalar_ts[4].shape)
    z_that_file = falkor_scalar_ts[4]
    z_location.append(z_that_file)    
    
    lat_that_file = np.empty(falkor_scalar_ts[2].shape)
    lat_that_file = falkor_scalar_ts[2]
    lat_location.append(lat_that_file)
    
    lon_that_file = np.empty(falkor_scalar_ts[3].shape)
    lon_that_file = falkor_scalar_ts[3]
    lon_location.append(lon_that_file)
    
    
    

    
temp_location = np.array(temp_location)
sal_location = np.array(sal_location)
z_location = np.array(z_location)
lat_location = np.array(lat_location)
lon_location = np.array(lon_location)

SA_falk_loc = np.empty_like(sal_location)
CT_falk_loc = np.empty_like(temp_location)
pressure_falk_loc = np.empty_like(z_location)
spic_falk_loc = np.empty_like(sal_location)


for i in np.arange(lat_location.shape[0]):
    
    pressure_falk_loc[i] = gsw.p_from_z(-z_location[i],lat_location[i])
    
    SA_falk_loc[i] = gsw.SA_from_SP(sal_location[i], pressure_falk_loc[i], lon_location[i], lat_location[i])
    
    CT_falk_loc[i] = gsw.CT_from_pt(sal_location[i], temp_location[i])
    
    spic_falk_loc[i] = gsw.spiciness0(SA_falk_loc[i], CT_falk_loc[i])




y = np.empty_like(lat_location)
x = np.empty_like(lat_location)


for i in np.arange(lat_location.shape[0]):
    y[i], x[i] = geo_tools.find_closest_model_point(
               lon_location[i],lat_location[i],lon_model,lat_model,tols={
        'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})
    
n = np.empty_like(lat_location)
m = np.empty_like(lat_location)


for i in np.arange(lat_location.shape[0]):
    n[i], m[i] = geo_tools.find_closest_model_point(
               lon_location[i],lat_location[i],lon,lat,tols={
        'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}})

y = y.astype(int)
x = x.astype(int)

n = n.astype(int)
m = m.astype(int)

sal_week = sal[112:123,:,:,:]
temp_week = temp[112:123,:,:,:]
spic_week = spic[112:123,:,:,:]
rho_week = spic[112:123,:,:,:]

sal_mean = np.mean(sal_week, axis=0)
temp_mean = np.mean(temp_week, axis=0)
spic_mean = np.mean(spic_week, axis=0)
rho_mean  = np.mean(rho_week, axis=0)


def add_model_data_to_bin_at_station_locations(file_number, z_bin_1, z_bin_2, z_bin_3,\
                    tem_bin_1, tem_bin_2, tem_bin_3, \
                   sal_bin_1, sal_bin_2, sal_bin_3, \
                   spic_bin_1, spic_bin_2, spic_bin_3):
    
    t = time_location[file_number]
    j = n[file_number]
    i = m[file_number]
        
    model_temp_ini = temp[t,:mbathy[j,i],j,i]
    model_temp = model_temp_ini[np.nonzero(model_temp_ini)]

    model_sal_ini = sal[t,:mbathy[j,i],j,i]
    model_sal = model_sal_ini[np.nonzero(model_sal_ini)]

    model_spice_ini = spic[t,:mbathy[j,i],j,i]
    model_spice = model_spice_ini[np.nonzero(model_sal_ini)]

    

    z_obs = np.array(z_location[file_number])

    
    
    
    func_temp = interp1d(zlevels[:model_temp.shape[0]], model_temp, fill_value='extrapolate')
    model_temp_interp = func_temp(z_obs)
    
    
    func_sal = interp1d(zlevels[:model_sal.shape[0]], model_sal, fill_value='extrapolate')
    model_sal_interp = func_sal(z_obs)
    
    func_spic = interp1d(zlevels[:model_spice.shape[0]], model_spice, fill_value='extrapolate')
    model_spice_interp = func_spic(z_obs)
    
        
    
    
       
    z_data_bin_1    = z_obs[np.where(z_obs < 60)]

    tem_data_bin_1  = model_temp_interp[np.where(z_obs < 60)]

    sal_data_bin_1  = model_sal_interp[np.where(z_obs < 60)]

    spic_data_bin_1 = model_spice_interp[np.where(z_obs < 60)]
    

    z_data_bin_2    = z_obs[np.where((z_obs >= 60) & (z_obs < 120))]

    tem_data_bin_2  = model_temp_interp[np.where((z_obs >= 60) & (z_obs < 120))]

    sal_data_bin_2  = model_sal_interp[np.where((z_obs >= 60) & (z_obs < 120))]

    spic_data_bin_2 = model_spice_interp[np.where((z_obs >= 60) & (z_obs < 120))]
    

    z_data_bin_3    = z_obs[np.where((z_obs >= 120)& (z_obs <=500))]

    tem_data_bin_3  = model_temp_interp[np.where((z_obs >= 120) & (z_obs <500))]

    sal_data_bin_3  = model_sal_interp[np.where((z_obs >= 120) & (z_obs <500))]

    spic_data_bin_3 = model_spice_interp[np.where((z_obs >= 120) & (z_obs <500))]
    
    
    z_bin_1 = np.append(arr= z_bin_1, values= z_data_bin_1)
    z_bin_2 = np.append(arr= z_bin_2, values= z_data_bin_2)
    z_bin_3 = np.append(arr= z_bin_3, values= z_data_bin_3)
    
    tem_bin_1 = np.append(arr= tem_bin_1, values= tem_data_bin_1)
    tem_bin_2 = np.append(arr= tem_bin_2, values= tem_data_bin_2)
    tem_bin_3 = np.append(arr= tem_bin_3, values= tem_data_bin_3)
    
    sal_bin_1 = np.append(arr= sal_bin_1, values= sal_data_bin_1)
    sal_bin_2 = np.append(arr= sal_bin_2, values= sal_data_bin_2)
    sal_bin_3 = np.append(arr= sal_bin_3, values= sal_data_bin_3)
    
    spic_bin_1 = np.append(arr= spic_bin_1, values= spic_data_bin_1)
    spic_bin_2 = np.append(arr= spic_bin_2, values= spic_data_bin_2)
    spic_bin_3 = np.append(arr= spic_bin_3, values= spic_data_bin_3)
    
    return z_bin_1, z_bin_2, z_bin_3, tem_bin_1, tem_bin_2, tem_bin_3, sal_bin_1, sal_bin_2, sal_bin_3, spic_bin_1, spic_bin_2, spic_bin_3
    

def add_data_to_bin(file_number, z_bin_1, z_bin_2, z_bin_3,\
                    tem_bin_1, tem_bin_2, tem_bin_3, \
                   sal_bin_1, sal_bin_2, sal_bin_3, \
                   spic_bin_1, spic_bin_2, spic_bin_3):
    



    z_data = z_location[file_number]

    tem_data = temp_location[file_number]

    sal_data = sal_location[file_number]

    spic_data = spic_falk_loc[file_number]
    
       
    z_data_bin_1    = z_data[np.where(z_data < 60)]

    tem_data_bin_1  = tem_data[np.where(z_data < 60)]

    sal_data_bin_1  = sal_data[np.where(z_data < 60)]

    spic_data_bin_1 = spic_data[np.where(z_data < 60)]
    

    z_data_bin_2    = z_data[np.where((z_data >= 60) & (z_data < 120))]

    tem_data_bin_2  = tem_data[np.where((z_data >= 60) & (z_data < 120))]

    sal_data_bin_2  = sal_data[np.where((z_data >= 60) & (z_data < 120))]

    spic_data_bin_2 = spic_data[np.where((z_data >= 60) & (z_data < 120))]
    

    z_data_bin_3    = z_data[np.where((z_data >= 120) & (z_data <500))]

    tem_data_bin_3  = tem_data[np.where((z_data >= 120)& (z_data <500))]

    sal_data_bin_3  = sal_data[np.where((z_data >= 120)& (z_data <500))]

    spic_data_bin_3 = spic_data[np.where((z_data >= 120)& (z_data <500))]
    
    
    z_bin_1 = np.append(arr= z_bin_1, values= z_data_bin_1)
    z_bin_2 = np.append(arr= z_bin_2, values= z_data_bin_2)
    z_bin_3 = np.append(arr= z_bin_3, values= z_data_bin_3)
    
    tem_bin_1 = np.append(arr= tem_bin_1, values= tem_data_bin_1)
    tem_bin_2 = np.append(arr= tem_bin_2, values= tem_data_bin_2)
    tem_bin_3 = np.append(arr= tem_bin_3, values= tem_data_bin_3)
    
    sal_bin_1 = np.append(arr= sal_bin_1, values= sal_data_bin_1)
    sal_bin_2 = np.append(arr= sal_bin_2, values= sal_data_bin_2)
    sal_bin_3 = np.append(arr= sal_bin_3, values= sal_data_bin_3)
    
    spic_bin_1 = np.append(arr= spic_bin_1, values= spic_data_bin_1)
    spic_bin_2 = np.append(arr= spic_bin_2, values= spic_data_bin_2)
    spic_bin_3 = np.append(arr= spic_bin_3, values= spic_data_bin_3)
    
    return z_bin_1, z_bin_2, z_bin_3, tem_bin_1, tem_bin_2, tem_bin_3, sal_bin_1, sal_bin_2, sal_bin_3, spic_bin_1, spic_bin_2, spic_bin_3
    

    
z_bin_1_m = []
z_bin_2_m = []
z_bin_3_m = []

tem_bin_1_m = []
tem_bin_2_m = []
tem_bin_3_m = []

sal_bin_1_m = []
sal_bin_2_m = []
sal_bin_3_m = []

spic_bin_1_m = []
spic_bin_2_m = []
spic_bin_3_m = []


for file_number in np.arange(1,len(file)-1):
    if file_number!=23:
        z_bin_1_m, z_bin_2_m, z_bin_3_m, \
        tem_bin_1_m, tem_bin_2_m,tem_bin_3_m, \
        sal_bin_1_m, sal_bin_2_m, sal_bin_3_m, \
        spic_bin_1_m, spic_bin_2_m, spic_bin_3_m = add_model_data_to_bin_at_station_locations(file_number, z_bin_1_m, z_bin_2_m, z_bin_3_m,\
                                                             tem_bin_1_m, tem_bin_2_m, tem_bin_3_m, \
                                                             sal_bin_1_m, sal_bin_2_m, sal_bin_3_m, \
                                                             spic_bin_1_m, spic_bin_2_m, spic_bin_3_m)
    
z_bin_1 = []
z_bin_2 = []
z_bin_3 = []

tem_bin_1 = []
tem_bin_2 = []
tem_bin_3 = []

sal_bin_1 = []
sal_bin_2 = []
sal_bin_3 = []

spic_bin_1 = []
spic_bin_2 = []
spic_bin_3 = []


for file_number in np.arange(1,len(file)-1):
    if file_number!=23:
        z_bin_1, z_bin_2, z_bin_3, \
        tem_bin_1, tem_bin_2,tem_bin_3, \
        sal_bin_1, sal_bin_2, sal_bin_3, \
        spic_bin_1, spic_bin_2, spic_bin_3 = add_data_to_bin(file_number, z_bin_1, z_bin_2, z_bin_3,\
                                                             tem_bin_1, tem_bin_2, tem_bin_3, \
                                                             sal_bin_1, sal_bin_2, sal_bin_3, \
                                                             spic_bin_1, spic_bin_2, spic_bin_3)

df =pd.DataFrame()
df['Observed_temp'] = np.concatenate((tem_bin_1, tem_bin_2, tem_bin_3), axis = 0)
df['Model_temp'] = np.concatenate((tem_bin_1_m, tem_bin_2_m, tem_bin_3_m), axis = 0)
df['Observed_Salinity'] = np.concatenate((sal_bin_1, sal_bin_2, sal_bin_3), axis = 0)
df['Model_Salinity'] = np.concatenate((sal_bin_1_m, sal_bin_2_m, sal_bin_3_m), axis = 0)
df['Observed_Spice'] = np.concatenate((spic_bin_1, spic_bin_2, spic_bin_3), axis = 0)
df['Model_Spice'] = np.concatenate((spic_bin_1_m, spic_bin_2_m, spic_bin_3_m), axis = 0)
df["Depth (m)"] = np.concatenate((z_bin_1[:], z_bin_2[:], z_bin_3[:]), axis = 0)


z = np.polyfit(df['Model_temp'],df['Observed_temp'], 2)
f = np.poly1d(z)

# calculate new x's and y's
x_new = np.arange(np.min(df['Model_temp']), np.max(df['Model_temp']), 0.01)
y_new = f(x_new)

x_fit = df['Model_temp']
y_fit = f(x_fit)

df['Fitted_temp'] = y_fit

changed_temp = np.empty_like(temp)
pressure_mean = gsw.p_from_z(-zlevels[:],np.mean(lat))
SA_mean_model = np.empty_like(temp)
CT_mean_model = np.empty_like(temp)
spic_mean_model = np.empty_like(temp)
rho_mean_model  = np.empty_like(temp)


for t in np.arange(112,125):
    for k in np.arange(temp.shape[1]):
        for j in np.arange(temp.shape[2]):
            for i in np.arange(temp.shape[3]):
                changed_temp[t,k,j,i] = f(temp[t,k,j,i])
                SA_mean_model[t,k,j,i] = gsw.SA_from_SP(sal[t,k,j,i], pressure_mean[k], lon[j,i], lat[j,i])
                CT_mean_model[t,k,j,i] = gsw.CT_from_pt(sal[t,k,j,i], changed_temp[t,k,j,i])
                spic_mean_model[t,k,j,i] = gsw.spiciness0(SA_mean_model[t,k,j,i],CT_mean_model[t,k,j,i])
                rho_mean_model[t,k,j,i] = gsw.density.rho(SA_mean_model[t,k,j,i], CT_mean_model[t,k,j,i], 0) 
                
path_to_save ='/data/ssahu/NEP36_Extracted_Months/'
bdy_file = nc.Dataset(path_to_save + 'NEP36_T_S_Spice_2013_Improved_temp_correction.nc', 'w', zlib=True);

bdy_file.createDimension('x', temp.shape[3]);
bdy_file.createDimension('y', temp.shape[2]);
bdy_file.createDimension('deptht', temp.shape[1]);
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
density   = bdy_file.createVariable('density', 'float32', ('time_counter', 'deptht', 'y', 'x'), zlib=True);



votemper[...]  = changed_temp[...];
vosaline[...]  = sal[...];
spiciness[...] = spic_mean_model[...];
density[...]   = rho_mean_model[...];


bdy_file.close()
    
print("The Model Corrected file is successfully written") 


print("End of Script: Thanks")



    
            
            
