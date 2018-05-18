import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d

NEP_aug = nc.Dataset('/home/ssahu/saurav/NEP36_T_S_Spice_aug.nc')


sal_aug = NEP_aug.variables['vosaline']
temp_aug = NEP_aug.variables['votemper']
spic_aug = NEP_aug.variables['spiciness']
rho_aug = NEP_aug.variables['density']

zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht']

mesh_mask = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/INV/mesh_mask.nc')
mbathy = mesh_mask['mbathy'][0,...]


NEP_jul = nc.Dataset('/home/ssahu/saurav/NEP36_T_S_Spice_july.nc')


sal_jul  = NEP_jul.variables['vosaline']
temp_jul = NEP_jul.variables['votemper']
spic_jul = NEP_jul.variables['spiciness']
rho_jul = NEP_jul.variables['density']

y_wcvi_slice = np.arange(230,350)
x_wcvi_slice = np.arange(550,650)




#znew = np.arange(0,150,0.1)
#dens_cont = np.arange(25.,27.,0.25/8.)
#tol = 0.001

#spic_iso = np.empty((rho_jul.shape[0],dens_cont.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
#rho_iso  =  np.empty((rho_jul.shape[0],dens_cont.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
#temp_iso = np.empty((rho_jul.shape[0],dens_cont.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
#sal_iso  =  np.empty((rho_jul.shape[0],dens_cont.shape[0],y_wcvi_slice.shape[0],y_wcvi_slice.shape[0]))



#t =12
znew = np.arange(0,250,0.05)

den = np.arange(23.,28.,0.1)
tol = 0.01


#rho_new = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))
#spic_new = np.empty((znew.shape[0],x_wcvi_slice.shape[0]))






#rho_0 = rho_jul[t,:,y_wcvi_slice,x_wcvi_slice] - 1000
#spic_0 = spic_jul[t,:,y_wcvi_slice,x_wcvi_slice]

spic_time_iso = np.empty((spic_jul.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
tem_time_iso  = np.empty((spic_jul.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
sal_time_iso  = np.empty((spic_jul.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))

for t in np.arange(spic_time_iso.shape[0]):
    
    rho_0  = rho_jul[t,:,y_wcvi_slice,x_wcvi_slice] - 1000
    spic_0 = spic_jul[t,:,y_wcvi_slice,x_wcvi_slice]
    tem_0  = temp_jul[t,:,y_wcvi_slice,x_wcvi_slice]
    sal_0  = sal_jul[t,:,y_wcvi_slice,x_wcvi_slice]

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


print("Calculating the depths of the isopycnals (in July) for 3D plots")

depth_rho_0 = np.empty((sal_time_iso[...].shape[0],sal_time_iso.shape[1],rho_jul.shape[2],rho_jul.shape[3]))

for t in np.arange(spic_time_iso.shape[0]):
    for iso in np.arange(den.shape[0]):
        for j in np.arange(230,350):
            for i in np.arange(550,650):
                if mbathy[j,i] > 0:
                    depth_rho_0[t,iso,j, i] = np.interp(den[iso], rho_jul[t,:mbathy[j, i], j, i]-1000, zlevels[:mbathy[j, i]])

depth_rho = np.empty_like(sal_time_iso[...])
depth_rho = depth_rho_0[:,:,y_wcvi_slice,x_wcvi_slice]

#for den in np.arange(dens_cont.shape[0]):
#    for t in np.arange(rho_jul.shape[0]):
#        for j in np.arange(y_wcvi_slice.shape[0]):
#            
#            for i in np.arange(y_wcvi_slice.shape[0]):
#                
#                print(i)

                #Choose the data slice in x-z
                
#                rho_0 = rho_jul[t,:,j,x_wcvi_slice] - 1000
#                spic_0 = spic_jul[t,:,j,x_wcvi_slice]
#                temp_0 = temp_jul[t,:,j,x_wcvi_slice]
#                sal_0 = sal_jul[t,:,j,x_wcvi_slice]
#                
#                # initialise the shapes of the variables#

#                rho_new = np.empty((znew.shape[0],rho_0.shape[1]))
#                spic_new = np.empty((znew.shape[0],rho_0.shape[1]))
#                temp_new = np.empty((znew.shape[0],rho_0.shape[1]))
#                sal_new = np.empty((znew.shape[0],rho_0.shape[1]))
#                ind = np.empty((znew.shape[0],rho_0.shape[1]))

                
                # Interpolate over z to choose the exact values of z for the isopycnals

#                f = interp1d(zlevels[:],rho_0[:,i],fill_value='extrapolate')
#                g = interp1d(zlevels[:],spic_0[:,i],fill_value='extrapolate')
#                h = interp1d(zlevels[:],temp_0[:,i],fill_value='extrapolate')
#                wine = interp1d(zlevels[:],sal_0[:,i],fill_value='extrapolate')
#                
#                # find the values of the variables at the fine z resolutions
#                
#
#                rho_new[:,i]  = f(znew[:])
#                spic_new[:,i] = g(znew[:])
#                temp_new[:,i] = h(znew[:])
#                sal_new[:,i]  = wine(znew[:])
#                
#                # find the indices which relate to those isopycnal values in x and z from a created boolean masked tuple ind 
#                
#                V = rho_new
#                ind = np.where((V>dens_cont[den]-tol)&(V<dens_cont[den]+tol))
#                
                # edit the intialised array with the values returned from the isopycnal indices
                
#                spic_iso[t,den,j,i] = spic_new[ind[0][:],ind[1][:]]
#                rho_iso[t,den,j,i]  = rho_new[ind[0][:],ind[1][:]]
#                temp_iso[t,den,j,i] = temp_new[ind[0][:],ind[1][:]]
#                sal_iso[t,den,j,i]  = sal_new[ind[0][:],ind[1][:]]
                

                
                

    
                                   
print("Writing the isopycnal data for July")                
                    
path_to_save = '/home/ssahu/saurav/'

bdy_file = nc.Dataset(path_to_save + 'NEP36_jul_along_isopycnal.nc', 'w', zlib=True);

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
zdepth_of_isopycnal = bdy_file.createVariable('Depth of Isopycnal', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)
#density = bdy_file.createVariable('density', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)





spiciness[...]           = spic_time_iso[...];
temperature[...]         = tem_time_iso[...];
salinity[...]            = sal_time_iso[...];
zdepth_of_isopycnal[...] = depth_rho[...]
#density[...]   = rho_iso[...];
isot[...] = den[:];
x[...] = x_wcvi_slice[:];
y[...] = y_wcvi_slice[:];

bdy_file.close()

print("File for July Written: Thanks")


print("Starting interpolation and data extraction for August")

spic_time_iso = np.empty((spic_aug.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
tem_time_iso  = np.empty((spic_aug.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))
sal_time_iso  = np.empty((spic_aug.shape[0],den.shape[0],y_wcvi_slice.shape[0],x_wcvi_slice.shape[0]))

for t in np.arange(spic_time_iso.shape[0]):
    
    rho_0  = rho_aug[t,:,y_wcvi_slice,x_wcvi_slice] - 1000
    spic_0 = spic_aug[t,:,y_wcvi_slice,x_wcvi_slice]
    tem_0  = temp_aug[t,:,y_wcvi_slice,x_wcvi_slice]
    sal_0  = sal_aug[t,:,y_wcvi_slice,x_wcvi_slice]

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


print("Calculating the depths of the isopycnals (in August) for 3D plots")

depth_rho_0 = np.empty((sal_time_iso[...].shape[0],sal_time_iso.shape[1],rho_jul.shape[2],rho_jul.shape[3]))

for t in np.arange(spic_time_iso.shape[0]):
    for iso in np.arange(den.shape[0]):
        for j in np.arange(230,350):
            for i in np.arange(550,650):
                if mbathy[j,i] > 0:
                    depth_rho_0[t,iso,j, i] = np.interp(den[iso], rho_aug[t,:mbathy[j, i], j, i]-1000, zlevels[:mbathy[j, i]])

depth_rho = np.empty_like(sal_time_iso[...])
depth_rho = depth_rho_0[:,:,y_wcvi_slice,x_wcvi_slice]                    
                    
print("Writing the isopycnal data for August")                
                    
path_to_save = '/home/ssahu/saurav/'

bdy_file = nc.Dataset(path_to_save + 'NEP36_aug_along_isopycnal.nc', 'w', zlib=True);

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
zdepth_of_isopycnal = bdy_file.createVariable('Depth of Isopycnal', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)
#density = bdy_file.createVariable('density', 'float32', ('time_counter','isot', 'y', 'x'), zlib=True)





spiciness[...]           = spic_time_iso[...];
temperature[...]         = temp_time_iso[...];
salinity[...]            = sal_time_iso[...];
zdepth_of_isopycnal[...] = depth_rho[...]
#density[...]   = rho_iso[...];
isot[...] = den[:];
x[...] = x_wcvi_slice[:];
y[...] = y_wcvi_slice[:];

bdy_file.close()

print("File for August Written: Thanks")
                    

    
    



