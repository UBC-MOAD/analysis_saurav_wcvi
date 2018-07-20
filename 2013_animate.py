import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
import matplotlib.cm as cm
from salishsea_tools import (nc_tools, gsw_calls, geo_tools, viz_tools)
import seabird
import cmocean as cmo
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

import matplotlib as mpl

print("The Modules were imported successfully")

bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')

mesh_mask = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/mesh_mask.nc')

mbathy = mesh_mask['mbathy'][0,180:350,480:650]

Z = bathy.variables['Bathymetry'][:]

y_wcvi_slice = np.arange(180,350)
x_wcvi_slice = np.arange(480,650)

zlevels = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d_20130429_20131025_grid_T_20130429-20130508.nc').variables['deptht']


lon = bathy['nav_lon'][...]
lat = bathy['nav_lat'][...]


lon_wcvi  = lon[180:350,480:650]
lat_wcvi  = lat[180:350,480:650]


NEP = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/NEP36_2013_T_S_Spice_larger_offshore_rho_correct.nc')


sal = NEP.variables['vosaline']
temp = NEP.variables['votemper']
spic = NEP.variables['spiciness']
rho = NEP.variables['density']


short_NEP_iso = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/2013_short_slice_NEP36_along_isopycnal_larger_offshore_rho_correct.nc')

short_spic_iso = short_NEP_iso.variables['spiciness']
short_iso_t = short_NEP_iso.variables['isot']

def plot_iso_den(t, rho_0):
    
    depth_rho_0 = np.zeros_like(sal[0,0,...])

    SPICE_on_rho_0 = np.zeros_like(depth_rho_0)

    for j in np.arange(y_wcvi_slice.shape[0]):
        for i in np.arange(x_wcvi_slice.shape[0]):
            if mbathy[j,i] > 0:
                depth_rho_0[j, i] = np.interp(rho_0, rho[t,:mbathy[j, i], j, i]-1000, zlevels[:mbathy[j, i]])


    k = np.where(short_iso_t[:] == rho_0)

    spic_tzyx = short_spic_iso[t,k[0],...]
    spic_tzyx[np.isnan(spic_tzyx)] = 0
    spic_iso = np.ma.masked_equal(spic_tzyx[0,...], 0)
        
    norm = mpl.colors.Normalize(vmin=-0.1,vmax=-0.05)


    plt.rcParams['contour.negative_linestyle'] = 'solid' # default is to have negative contours with dashed lines
    plt.rcParams.update({'font.size':16})




    cmap = plt.get_cmap(cmo.cm.balance)
    cmap.set_bad('Burlywood')

    V_normalized = (spic_iso - spic_iso.min().min())
    V_normalized = V_normalized / V_normalized.max().max()
    spic_plot = V_normalized


    ls = LightSource(290, 35)
    #     cmap1 = ls.shade(spic_iso, cmap=cmap, vert_exag=0.8, vmin=-0.4, vmax =-0.15, blend_mode='overlay')

    cmap1 = ls.shade(spic_iso, cmap=cmap, vert_exag=0.8, vmin=-0.1, vmax =-0.05, blend_mode='overlay')




    fig = plt.figure(figsize=(25, 20))
    ax = fig.gca(projection='3d')
    
    if t <= 1:
        month = 'April'
        ax.set_title('Spiciness on {0} April 2013, at isopycnal level of \u2248 {d:.2f} '.format(t+29, d=rho_0))
            
    if 1 < t <=32:
        month = 'May'
        ax.set_title('Spiciness on {0} May 2013, at isopycnal level of \u2248 {d:.2f} '.format(t-1, d=rho_0))
        
    if 32 < t <= 62:
        month = 'June'
        ax.set_title('Spiciness on {0} June 2013, at isopycnal level of \u2248 {d:.2f} '.format(t-32, d=rho_0))
        
    if 62 < t <= 93:
        month == 'July'
        ax.set_title('Spiciness on {0} July 2013, at isopycnal level of \u2248 {d:.2f} '.format(t-62, d=rho_0))
        
    if 83 < t <= 124:
        month = 'August'
        ax.set_title('Spiciness on {0} August 2013, at isopycnal level of \u2248 {d:.2f} '.format(t-83, d=rho_0))
        
    if 114 < t <= 154:
        month = 'September'
        ax.set_title('Spiciness on {0} September 2013, at isopycnal level of \u2248 {d:.2f} '.format(t-114, d=rho_0))
    
    if 154 < t <= 185:
        month = 'October'
        ax.set_title('Spiciness on {0} October 2013, at isopycnal level of \u2248 {d:.2f} '.format(t-154, d=rho_0))
    
    X, Y = np.meshgrid(x_wcvi_slice[:],y_wcvi_slice[:])
    #     surf = ax.plot_surface(np.flip(Y, axis=0), X, -depth_rho_0[180:350,480:650], facecolors=cmap1,linewidth=0, antialiased=False, rstride=1, cstride=1)
    surf = ax.plot_surface(lon_wcvi, lat_wcvi,-depth_rho_0[:,:], facecolors=cmap1, linewidth=0, antialiased=False, rstride=1, cstride=1)
    ax.set_aspect('auto')

    lb8_VecStart_x = lon[264, 599]
    lb8_VecStart_y = lat[264, 599]
    lb8_VecStart_z = 0
    lb8_VecEnd_x = lon[264, 599]
    lb8_VecEnd_y = lat[264, 599]
    lb8_VecEnd_z  =-150

    E01_VecStart_x = lon[322, 553]
    E01_VecStart_y = lat[322, 553]
    E01_VecStart_z = -13
    E01_VecEnd_x = lon[322, 553]
    E01_VecEnd_y = lat[322, 553]
    E01_VecEnd_z  = -85

    A1_VecStart_x = lon[269, 572]
    A1_VecStart_y = lat[269, 572]
    A1_VecStart_z = -84
    A1_VecEnd_x = lon[269, 572]
    A1_VecEnd_y = lat[269, 572]
    A1_VecEnd_z  = -200

    ax.plot(xs = np.array([lb8_VecStart_x, lb8_VecEnd_x], dtype=float), ys  = np.array([lb8_VecStart_y, lb8_VecEnd_y], dtype=float), zs=np.array([lb8_VecStart_z, lb8_VecEnd_z], dtype=float), label = 'LB08')

    ax.plot(xs = np.array([E01_VecStart_x, E01_VecEnd_x], dtype=float), ys  = np.array([E01_VecStart_y, E01_VecEnd_y], dtype=float), zs=np.array([E01_VecStart_z, E01_VecEnd_z], dtype=float), label = 'E01')

    ax.plot(xs = np.array([A1_VecStart_x, A1_VecEnd_x], dtype=float), ys  = np.array([A1_VecStart_y, A1_VecEnd_y], dtype=float), zs=np.array([A1_VecStart_z, A1_VecEnd_z], dtype=float), label = 'A1')  
    ax.set_ylabel('Latitude', fontsize = 18, labelpad= 18)
    ax.set_xlabel('Longitude', fontsize = 18, labelpad= 18)
    ax.set_zlabel('Depth (m)', fontsize = 18, labelpad= 18)
    ax.set_zlim(-250,0)
    m = cm.ScalarMappable(cmap=plt.get_cmap(cmo.cm.balance))
    m.set_array(spic_iso)
    #     m.set_clim(-0.4, -0.05)
    m.set_clim(-0.1, -0.05)
    plt.colorbar(m)
    ax.set_aspect('auto')
    ax.legend(loc='best', fancybox=True, framealpha=0.25)
    ax.view_init(35, 240) # elevation and azimuth
    
    plt.savefig('/home/ssahu/saurav/3D_images_for_video_spice/rho_26_4_{0}.png'.format(t))
    plt.close()


rho_0 = 26.4
for t in np.arange(rho.shape[0]):
    plot_iso_den(t, rho_0)
    