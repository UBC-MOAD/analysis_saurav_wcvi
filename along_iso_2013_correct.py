import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d

zlevels = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d_20130429_20131025_grid_T_20130429-20130508.nc').variables['deptht']

mesh_mask = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/mesh_mask.nc')
mbathy = mesh_mask['mbathy'][0,...]


#NEP_2013 = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/NEP36_2013_T_S_Spice_larger_offshore_rho_correct.nc')

NEP_2013 = nc.Dataset('/data/ssahu/NEP36_Extracted_Months/NEP36_2013_T_S_Spice_larger_offshore_rho_correct_\
including_spring_parallel.nc')

sal  = NEP_2013.variables['vosaline']
temp = NEP_2013.variables['votemper']
spic = NEP_2013.variables['spiciness']
rho = NEP_2013.variables['density']

y_wcvi_slice = np.arange(180,350)
x_wcvi_slice = np.arange(480,650)


znew = np.arange(0,250,0.1)

den = np.arange(26.,26.6,0.1)
tol = 0.01


spic_iso = np.empty((rho.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
rho_iso  =  np.empty((rho.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
temp_iso = np.empty((rho.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
sal_iso  =  np.empty((rho.shape[0],den.shape[0],y_wcvi_slice.shape[0],y_wcvi_slice.shape[0]))





rho_new = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
spic_new = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))






spic_time_iso = np.empty((spic.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
tem_time_iso  = np.empty((spic.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
sal_time_iso  = np.empty((spic.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))

for t in np.arange(spic_time_iso.shape[0]):
    
    print(t)
    
    rho_0  = rho[t,:,:,:] - 1000
    spic_0 = spic[t,:,:,:]
    tem_0  = temp[t,:,:,:]
    sal_0  = sal[t,:,:,:]

    spic_spec_iso = np.empty((den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
    tem_spec_iso  = np.empty((den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
    sal_spec_iso  = np.empty((den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))

    for iso in np.arange(den.shape[0]):

        spic_den = np.empty((y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
        tem_den  = np.empty((y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
        sal_den  = np.empty((y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))

        for j in np.arange(y_wcvi_slice.shape[0]):
    
    
            spic_iso = np.empty(x_wcvi_slice.shape[0])
            sal_iso  = np.empty(x_wcvi_slice.shape[0])
            tem_iso  = np.empty(x_wcvi_slice.shape[0])
     
        
            rho_new  = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
            spic_new = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
            tem_new  = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
            sal_new  = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
    
            for i in np.arange(rho_new.shape[1]):

    
                f = interp1d(zlevels[:],rho_0[:,j,i],fill_value='extrapolate')
                g = interp1d(zlevels[:],spic_0[:,j,i],fill_value='extrapolate')
                h = interp1d(zlevels[:],tem_0[:,j,i],fill_value='extrapolate')
                p = interp1d(zlevels[:],sal_0[:,j,i],fill_value='extrapolate')


                rho_new[:,i]  = f(znew[:])
                spic_new[:,i] = g(znew[:])
                tem_new[:,i]  = h(znew[:])
                sal_new[:,i]  = p(znew[:])

                V = rho_new[:,i]

                ind = (V>den[iso]-tol)&(V<den[iso]+tol)

                spic_iso[i] = np.nanmean(spic_new[ind,i])
                tem_iso[i] = np.nanmean(tem_new[ind,i])
                sal_iso[i] = np.nanmean(sal_new[ind,i])
    
                spic_den[j,i] = spic_iso[i]
                tem_den[j,i]  = tem_iso[i]
                sal_den[j,i]  = sal_iso[i]
        
                spic_spec_iso[iso,j,i] = spic_den[j,i]
                tem_spec_iso[iso,j,i]  = tem_den[j,i]
                sal_spec_iso[iso,j,i]  = sal_den[j,i]
        
                spic_time_iso[t,iso,j,i] = spic_spec_iso[iso,j,i]
                tem_time_iso[t,iso,j,i]  = tem_spec_iso[iso,j,i]
                sal_time_iso[t,iso,j,i]  = sal_spec_iso[iso,j,i]



                
                

    
                                   
print("Writing the isopycnal data")                
                    
path_to_save = '/data/ssahu/NEP36_Extracted_Months/'

bdy_file = nc.Dataset(path_to_save + '2013_NEP36_along_isopycnal_larger_offshore_rho_correct_including_spring.nc', 'w', zlib=True);

bdy_file.createDimension('x', spic_time_iso.shape[3]);
bdy_file.createDimension('y', spic_time_iso.shape[2]);
bdy_file.createDimension('isot', spic_time_iso.shape[1]);
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


spiciness = bdy_file.createVariable('spiciness', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)
temperature = bdy_file.createVariable('temperature', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)
salinity = bdy_file.createVariable('salinity', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)
#zdepth_of_isopycnal = bdy_file.createVariable('Depth of Isopycnal', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)
#density = bdy_file.createVariable('density', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)





spiciness[...]           = spic_time_iso[...];
temperature[...]         = tem_time_iso[...];
salinity[...]            = sal_time_iso[...];
#zdepth_of_isopycnal[...] = depth_rho[...]
#density[...]   = rho_iso[...];
isot[...] = den[:];
x[...] = x_wcvi_slice[:];
y[...] = y_wcvi_slice[:];

bdy_file.close()

print("File Written: Thanks")



