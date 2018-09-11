import numpy as np
import netCDF4 as nc


bathy = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/Bathymetry_EastCoast_NEMO_R036_GEBCO_corr_v14.nc')


Z = bathy.variables['Bathymetry']

y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))

z_wcvi = Z[y_wcvi_slice, x_wcvi_slice]
lon_wcvi = bathy['nav_lon'][180:350, 480:650]
lat_wcvi = bathy['nav_lat'][180:350, 480:650]

slice_file = nc.Dataset('/data/ssahu/WCVI_sliced_bathy_NEP36_original.nc', 'w', zlib=True);

slice_file.createDimension('xdim', z_wcvi.shape[1]);
slice_file.createDimension('ydim', z_wcvi.shape[0]);


xdim = slice_file.createVariable('xdim', 'int32', ('xdim',), zlib=True);
xdim.units = 'indices';
xdim.longname = 'x indices';

ydim = slice_file.createVariable('ydim', 'int32', ('ydim',), zlib=True);
ydim.units = 'indices';
ydim.longname = 'y indices';



bathymetry = slice_file.createVariable('Bathymetry', 'float32', ('ydim', 'xdim'), zlib=True);
bathymetry[...] = z_wcvi[...];

nav_lon = slice_file.createVariable('nav_lon', 'float32', ('ydim', 'xdim'), zlib=True);
nav_lon[...] = lon_wcvi[...];

nav_lat = slice_file.createVariable('nav_lat', 'float32', ('ydim', 'xdim'), zlib=True); 
nav_lat[...] = lat_wcvi[...];

slice_file.close()

print("The script has run to completion: Thanks")


