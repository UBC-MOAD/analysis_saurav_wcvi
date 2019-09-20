from __future__ import division
import glob
import os
import fnmatch
from collections import namedtuple, OrderedDict
import pandas as pd
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

import seaborn as sns
from windrose import plot_windrose
from windrose import WindroseAxes


from dateutil        import parser
from datetime import datetime


print("The Modules were imported successfully")

def find_principal_axis_of_dataset_in_degrees(u_data,v_data):
    
    u_perturb = u_data - np.mean(u_data)
    v_perturb = v_data - np.mean(v_data)
    
    coords = np.vstack([u_perturb, v_perturb])

    cov = np.cov(coords)
    
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    
    theta = np.tanh((x_v1)/(y_v1))  
#     rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
#                       [np.sin(theta), np.cos(theta)]])
#     transformed_mat = rotation_mat * coords
#     # plot the transformed blob
#     x_transformed, y_transformed = transformed_mat.A

    return np.rad2deg(theta)

def find_major_eigenvectors(u_data,v_data):
    
    u_data = u_data[~np.isnan(u_data)]
    v_data = v_data[~np.isnan(v_data)]
    
    u_perturb = u_data - np.mean(u_data)
    v_perturb = v_data - np.mean(v_data)
    
    coords = np.vstack([u_perturb, v_perturb])

    cov = np.cov(coords)
    
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]
    
    return x_v1, y_v1, x_v2, y_v2


    
def U_timeseries_at_WCVI_locations(grid_U):
    
    u_vel = grid_U.variables['uo'][:,:,:,:]

    
    vector_u = namedtuple('vector_u', 'u_vel')

    return vector_u(u_vel)


def V_timeseries_at_WCVI_locations(grid_V):
    
    v_vel = grid_V.variables['vo'][:,:,:,:]

    
    vector_v = namedtuple('vector_v', 'v_vel')

    return vector_v(v_vel)




def uv_wind_timeseries_at_point(grid_weather, j, i, datetimes=False):
    """Return the u and v wind components and time counter values
    at a single grid point from a weather forcing dataset.

    :arg grid_weather: Weather forcing dataset, typically from an
                       :file:`ops_yYYYYmMMdDD.nc` file produced by the
                       :py:mod:`nowcast.workers.grid_to_netcdf` worker.
    :type grid_weather: :py:class:`netCDF4.Dataset`

    :arg int j: j-direction (longitude) index of grid point to get wind
                components at.

    :arg int i: i-direction (latitude) index of grid point to get wind
                components at.

    :arg boolean datetimes: Return time counter values as
                            :py:class:`datetime.datetime` objects if
                            :py:obj:`True`, otherwise return them as
                            :py:class:`arrow.Arrow` objects (the default).

    :returns: 2-tuple of 1-dimensional :py:class:`numpy.ndarray` objects,
              The :py:attr:`u` attribute holds the u-direction wind
              component,
              The :py:attr:`v` attribute holds the v-direction wind
              component,
              and the :py:attr:`time` attribute holds the time counter
              values.
    :rtype: :py:class:`collections.namedtuple`
    """
    u_wind = grid_weather.variables['u_wind'][:, j, i]
    v_wind = grid_weather.variables['v_wind'][:, j, i]
    time = timestamp(grid_weather, range(len(u_wind)))
    if datetimes:
        time = np.array([a.datetime for a in time])
    wind_ts = namedtuple('wind_ts', 'u, v, time')

    return wind_ts(u_wind, v_wind, np.array(time))

def timestamp(dataset, tindex, time_var='time_counter'):
    """Return the time stamp of the tindex time_counter value(s) in dataset.

    The time stamp is calculated by adding the time_counter[tindex] value
    (in seconds) to the dataset's time_counter.time_origin value.

    :arg dataset: netcdf dataset object
    :type dataset: :py:class:`netCDF4.Dataset`

    :arg tindex: time_counter variable index.
    :type tindex: int or iterable

    :arg time_var: name of the time variable
    :type time_var: str

    :returns: Time stamp value(s) at tindex in the dataset.
    :rtype: :py:class:`Arrow` instance or list of instances
    """
    time_orig = time_origin(dataset, time_var=time_var)
    time_counter = dataset.variables[time_var]
    try:
        iter(tindex)
    except TypeError:
        tindex = [tindex]
    results = []
    for i in tindex:
        try:
            results.append(time_orig + timedelta(seconds=time_counter[i]))
        except IndexError:
            raise IndexError(
                'time_counter variable has no tindex={}'.format(tindex))
    if len(results) > 1:
        return results
    else:

        return results[0]

def time_origin(dataset, time_var='time_counter'):
    """Return the time_var.time_origin value.

    :arg dataset: netcdf dataset object
    :type dataset: :py:class:`netCDF4.Dataset` or :py:class:`xarray.Dataset`

    :arg time_var: name of time variable
    :type time_var: str

    :returns: Value of the time_origin attribute of the time_counter
              variable.
    :rtype: :py:class:`Arrow` instance
    """
    try:
        time_counter = dataset.variables[time_var]
    except KeyError:
        raise KeyError(
            'dataset does not have {time_var} variable'.format(
                time_var=time_var))
    try:
        # netCDF4 dataset
        time_orig = time_counter.time_origin.title()
    except AttributeError:
        try:
            # xarray dataset
            time_orig = time_counter.attrs['time_origin'].title()
        except KeyError:
            raise AttributeError(
                'NetCDF: '
                '{time_var} variable does not have '
                'time_origin attribute'.format(time_var=time_var))
    value = arrow.get(
        time_orig,
        ['YYYY-MMM-DD HH:mm:ss',
         'DD-MMM-YYYY HH:mm:ss',
         'YYYY-MM-DD HH:mm:ss'])

    return value

import numpy

def smooth(x,window_len=24,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s, mode='same')
    
    return y[window_len - 1:-window_len + 1]

    
def df_derived_by_shift(df,lag=0,NON_DER=[]):
    df = df.copy()
    if not lag:
        return df
    cols ={}
    for i in np.arange(1,lag+1):
        for x in list(df.columns):
            if x not in NON_DER:
                if not x in cols:
                    cols[x] = ['{}_{}'.format(x, i)]
                else:
                    cols[x].append('{}_{}'.format(x, i))
    for k,v in cols.items():
        columns = v
        dfn = pd.DataFrame(data=None, columns=columns, index=df.index)    
        i = 1
        for c in columns:
            dfn[c] = df[k].shift(periods=i)
            i+=1
        df = pd.concat([df, dfn], axis=1, join_axes=[df.index])
    return df









"""
Constants
=========
"""

# ---- physical constants
g = 9.8  # 9.7803 in COARE FIXME: change to sw.grav
""" acceleration due to gravity [m s :sup:`-2`] """

sigmaSB = 5.6697e-8
""" Stefan-Boltzmann constant [W m :sup:`-2` K :sup:`-4`] """

eps_air = 0.62197
""" Molecular weight ratio [water air :sup:`-1`] """

CtoK = 273.15  # 273.16
""" Conversion factor for [:math:`^\\circ` C] to [:math:`^\\circ` K] """

gas_const_R = 287.04  # NOTE: 287.1 in COARE
""" Gas constant for dry air [J kg :sup:`-1` K :sup:`-1`] """


# ---- meteorological constants
kappa = 0.4  # NOTE: 0.41
""" von Karman's constant """

charn = Charnock_alpha = 0.011  # NOTE: 0.018
""" Charnock constant. For determining roughness length at sea given friction
velocity, used in Smith formulas for drag coefficient and also in Fairall and
Edson. Ese alpha = 0.011 for open-ocean and alpha = 0.018 for fetch-limited
(coastal) regions."""

R_roughness = 0.11
""" limiting roughness Reynolds for aerodynamically smooth flow """


# ---- defaults suitable for boundary-layer studies
cp = 1004.7
""" heat capacity of air [J kg :sup:`-1` K :sup:`-1`] """

Qsat_coeff = 0.98
""" saturation specific humidity coefficient reduced by 2% over salt water """


# ------ short-wave flux calculations
Solar_const = 1368.0
""" The solar constant [W/m^2] represents a mean of satellite measurements
made over the last sunspot cycle (1979-1995) taken from Coffey et al (1995),
Earth System Monitor, 6, 6-10."""


# ---- long-wave flux calculations
emiss_lw = 0.985
""" long-wave emissivity of ocean from Dickey et al (1994), J. Atmos. Oceanic
Tech., 11, 1057-1076."""

# Default values
P_default = 1020.





def stress(sp, z=5., drag='largepond', rho_air=1.22, Ta=10.):
    """Computes the neutral wind stress.
    Parameters
    ----------
    sp : array_like
         wind speed [m s :sup:`-1`]
    z : float, array_like, optional
        measurement height [m]
    rho_air : array_like, optional
           air density [kg m :sup:`-3`]
    drag : str
           neutral drag by:
           'largepond' <-- default
           'smith'
           'vera'
    Ta : array_like, optional
         air temperature [:math:`^\\circ` C]
    Returns
    -------
    tau : array_like
          wind stress  [N m :sup:`-2`]
    See Also
    --------
    cdn
    Examples
    --------
    >>> from airsea import windstress as ws
    >>> ws.stress([10., 0.2, 12., 20., 30., 50.], 10)
    array([  1.40300000e-01,   5.61200000e-05,   2.23113600e-01,
             8.73520000e-01,   2.67912000e+00,   1.14070000e+01])
    >>> kw = dict(rho_air=1.02, Ta=23.)
    >>> ws.stress([10., 0.2, 12., 20., 30., 50.], 15, 'smith', **kw)
    array([  1.21440074e-01,   5.32531576e-05,   1.88322389e-01,
             6.62091968e-01,   1.85325310e+00,   7.15282267e+00])
    >>> ws.stress([10., 0.2, 12., 20., 30., 50.], 8, 'vera')
    array([  1.50603698e-01,   7.16568379e-04,   2.37758830e-01,
             9.42518454e-01,   3.01119044e+00,   1.36422742e+01])
    References
    ----------
    .. [1] Large and Pond (1981), J. Phys. Oceanog., 11, 324-336.
    .. [2] Smith (1988), J. Geophys. Res., 93, 311-326.
    .. [3] E. Vera (1983) FIXME eqn. 8 in Large, Morzel, and Crawford (1995),
    J. Phys. Oceanog., 25, 2959-2971.
    Modifications: Original from AIR_SEA TOOLBOX, Version 2.0
    03-08-1997: version 1.0
    08-26-1998: version 1.1 (revised by RP)
    04-02-1999: versin 1.2 (air density option added by AA)
    08-05-1999: version 2.0
    11-26-2010: Filipe Fernandes, Python translation.
    """
    z, sp = np.asarray(z), np.asarray(sp)
    Ta, rho_air = np.asarray(Ta), np.asarray(rho_air)

    # Find cd and ustar.
    if drag == 'largepond':
        cd, sp = cdn(sp, z, 'largepond')
    elif drag == 'smith':
        cd, sp = cdn(sp, z, 'smith', Ta)
    elif drag == 'vera':
        cd, sp = cdn(sp, z, 'vera')
    else:
        print('Unknown method')  # FIXME: raise a proper python error

    tau = rho_air * (cd * sp ** 2)

    return tau


def cdn(sp, z, drag='largepond', Ta=10):
    """Computes neutral drag coefficient.
    Methods available are: Large & Pond (1981),  Vera (1983) or Smith (1988)
    Parameters
    ----------
    sp : array_like
         wind speed [m s :sup:`-1`]
    z : float, array_like
        measurement height [m]
    drag : str
           neutral drag by:
           'largepond' <-- default
           'smith'
           'vera'
    Ta : array_like, optional for drag='smith'
         air temperature [:math:`^\\circ` C]
    Returns
    -------
    cd : float, array_like
         neutral drag coefficient at 10 m
    u10 : array_like
          wind speed at 10 m [m s :sup:`-1`]
    See Also
    --------
    stress, spshft, visc_air
    Notes
    -----
    Vera (1983): range of fit to data is 1 to 25 [m s :sup:`-1`].
    Examples
    --------
    >>> from airsea import windstress as ws
    >>> ws.cdn([10., 0.2, 12., 20., 30., 50.], 10)
    (array([ 0.00115,  0.00115,  0.00127,  0.00179,  0.00244,  0.00374]),
     array([ 10. ,   0.2,  12. ,  20. ,  30. ,  50. ]))
    >>> ws.cdn([10., 0.2, 12., 20., 30., 50.], 15, 'vera')
    (array([ 0.00116157,  0.01545237,  0.00126151,  0.00174946,  0.00242021,
            0.00379521]),
     array([  9.66606155,   0.17761896,  11.58297824, 19.18652915,
            28.5750255 ,  47.06117334]))
    >>> ws.cdn([10., 0.2, 12., 20., 30., 50.], 20, 'smith', 20.)
    (array([ 0.00126578,  0.00140818,  0.00136533,  0.00173801,  0.00217435,
            0.00304636]),
     array([  9.41928554,   0.18778865,  11.27787697,  18.65250005,
            27.75712916,  45.6352786 ]))
    References
    ----------
    .. [1] Large and Pond (1981), J. Phys. Oceanog., 11, 324-336.
    .. [2] Smith (1988), J. Geophys. Res., 93, 311-326.
    .. [3] E. Vera (1983) FIXME eqn. 8 in Large, Morzel, and Crawford (1995),
    J. Phys. Oceanog., 25, 2959-2971.
    Modifications: Original from AIR_SEA TOOLBOX, Version 2.0
    03-08-1997: version 1.0
    08-26-1998: version 1.1 (vectorized by RP)
    08-05-1999: version 2.0
    11-26-2010: Filipe Fernandes, Python translation.
    """
    # convert input to numpy array
    sp, z, Ta = np.asarray(sp), np.asarray(z), np.asarray(Ta)

    tol = 0.00001  # Iteration end point.

    if drag == 'largepond':
        a = np.log(z / 10.) / kappa  # Log-layer correction factor.
        u10o = np.zeros(sp.shape)
        cd = 1.15e-3 * np.ones(sp.shape)
        u10 = sp / (1 + a * np.sqrt(cd))
        ii = np.abs(u10 - u10o) > tol

        while np.any(ii):
            u10o = u10
            cd = (4.9e-4 + 6.5e-5 * u10o)  # Compute cd(u10).
            cd[u10o < 10.15385] = 1.15e-3
            u10 = sp / (1 + a * np.sqrt(cd))  # Next iteration.
            # Keep going until iteration converges.
            ii = np.abs(u10 - u10o) > tol

    elif drag == 'smith':
        visc = visc_air(Ta)

        # Remove any sp==0 to prevent division by zero
        # i = np.nonzero(sp == 0)
        # sp[i] = 0.1 * np.ones(len(i)) FIXME

        # initial guess
        ustaro = np.zeros(sp.shape)
        ustarn = 0.036 * sp

        # iterate to find z0 and ustar
        ii = np.abs(ustarn - ustaro) > tol
        while np.any(ii):
            ustaro = ustarn
            z0 = Charnock_alpha * ustaro ** 2 / g + R_roughness * visc / ustaro
            ustarn = sp * (kappa / np.log(z / z0))
            ii = np.abs(ustarn - ustaro) > tol

        sqrcd = kappa / np.log(10. / z0)
        cd = sqrcd ** 2
        u10 = ustarn / sqrcd
    elif drag == 'vera':
        # constants in fit for drag coefficient
        A = 2.717e-3
        B = 0.142e-3
        C = 0.0764e-3

        a = np.log(z / 10.) / kappa  # Log-layer correction factor.
        # Don't start iteration at 0 to prevent blowups.
        u10o = np.zeros(sp.shape) + 0.1
        cd = A / u10o + B + C * u10o
        u10 = sp / (1 + a * np.sqrt(cd))

        ii = np.abs(u10 - u10o) > tol
        while np.any(ii):
            u10o = u10
            cd = A / u10o + B + C * u10o
            u10 = sp / (1 + a * np.sqrt(cd))  # Next iteration.
            # Keep going until iteration converges.
            ii = np.abs(u10 - u10o) > tol
    else:
        print('Unknown method')  # FIXME: raise a proper python error.

    return cd, u10


def get_alongshore_currents(u_curr, v_curr):
    
    pri_angle    = find_principal_axis_of_dataset_in_degrees(u_data=u_curr, v_data=v_curr)
    pri_angle    = np.radians(pri_angle)
#     direc        = 90 - np.rad2deg(np.arctan(v_curr/u_curr))
#     Speed_curr   = np.sqrt(u_curr**2 + v_curr**2)
    
    coords = np.vstack([u_curr, v_curr])
    
    rotation_mat = np.matrix([[np.cos(pri_angle), -np.sin(pri_angle)],
                  [np.sin(pri_angle), np.cos(pri_angle)]])
    transformed_mat = rotation_mat * coords
    
    alongshore_rot = transformed_mat[1,:]
    
    alongshore_rot = alongshore_rot.tolist()
    
    
    
    
    
#     dir_new      = direc - pri_angle
    
#     alongshore_rot = np.empty_like(dir_new)
#     alongshore_rot[:] = np.multiply(Speed_curr[:],np.sin(np.deg2rad(dir_new)))
    
    return alongshore_rot[0]

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]
    
    
def calculate_strength_of_adcp_undercurrent(t, adcp_alongshore, adcp_depths):
       
    ind  = np.array(np.where(adcp_alongshore[:] > 0))
    
    if (ind.size == 0) == False:

        mean_undercurr = np.mean(adcp_alongshore[ind])

        if ind[0][0] !=  ind[0][-1]:

            depths_mean = np.mean(adcp_depths[ind[0][0]:ind[0][-1]])

            strength = np.trapz(adcp_alongshore[ind[0][0]:ind[0][-1]], adcp_depths[ind[0][0]:ind[0][-1]])


        else:
            depths_mean = adcp_depths[ind[0][0]]
            
    else:
        strength = 0

#     strength = np.multiply(mean_undercurr,depths_mean)
        
    return strength



def calculate_strength_of_model_undercurrent(t, model_alongshore, model_depths):

    ind  = np.array(np.where(model_alongshore[20:] > 0))
    
    if (ind.size == 0) == False:
        
        if ind[0][0] !=  ind[0][-1]:

            mean_undercurr = np.mean(model_alongshore[20+ind[0][0]:20+ind[0][-1]])

            depths_mean = np.absolute(np.mean(zlevels[20+ind[0][0]:20+ind[0][-1]]))

            strength = np.trapz(model_alongshore[20+ind[0][0]:20+ind[0][-1]], zlevels[20+ind[0][0]:20+ind[0][-1]])
            
            if np.ma.is_masked(strength) ==  True:
            
                strength = np.multiply(depths_mean, mean_undercurr)

        else:
            depths_mean = np.absolute(zlevels[20+ind[0][0]])
            mean_undercurr = np.absolute(model_alongshore[20+ind[0][0]])
            strength = np.multiply(depths_mean, mean_undercurr)
            
    else:
        strength = 0

#     strength = np.multiply(mean_undercurr,depths_mean)
        
    return strength

print("The functions are defined")

newport_data = pd.read_table('/data/ssahu/Falkor_2013/winds/Newport_winds_unfiltered.csv', delim_whitespace=1, parse_dates= True, header = None, skiprows=2)


df_cut = newport_data.drop(newport_data.columns[7:], axis=1)
columns = ['Year','Month','Day','Hour','Min','Wind Direction (deg)','Wind Speed','GDR','GST','GTIME']
effective = columns[:7]
df_cut.columns = effective


df_cut['Year']   = df_cut['Year'].astype(str)
df_cut['Day']    = df_cut['Day'].astype(str)
df_cut['Month']  = df_cut['Month'].astype(str)
df_cut['Hour']   = df_cut["Hour"].astype(str)

df_cut.columns = df_cut.columns.str.replace(' ', '')



df_cut['Datetime'] = df_cut['Year'] + "-" + df_cut['Month'] + "-" + df_cut['Day']+ " " + df_cut["Hour"] + ":00:00"

df_cut["Datetime"] = pd.to_datetime(df_cut["Datetime"])


df_cut = df_cut.iloc[:,4:]

df_cut = df_cut.set_index(pd.DatetimeIndex(df_cut["Datetime"]))

df_subset = df_cut.loc['2013-02-28 01:00:00':'2013-08-31 23:00:00']

df_subset = df_subset.resample('60T').mean().reset_index()


newport_datetime = df_subset['Datetime']

ws_newport = df_subset['WindSpeed']
wd_newport = df_subset['WindDirection(deg)']


u_wind_newport = np.multiply(ws_newport[:],np.cos(wd_newport))

v_wind_newport = np.multiply(ws_newport[:],np.sin(wd_newport))

df_subset_daily = df_cut.loc['2013-02-28 01:00:00':'2013-08-31 23:00:00']

df_subset_daily = df_subset_daily.resample('D').mean().reset_index()

ws_newport_daily = df_subset_daily['WindSpeed']
wd_newport_daily = df_subset_daily['WindDirection(deg)']


u_wind_newport_daily = np.multiply(ws_newport_daily[:],np.cos(wd_newport_daily))

v_wind_newport_daily = np.multiply(ws_newport_daily[:],np.sin(wd_newport_daily))

La_Peruse_data = pd.read_table('/data/ssahu/Falkor_2013/winds/La_persue_wave_buoy.csv',delim_whitespace=1,header = None, skiprows=7)

a = La_Peruse_data[0].str.split(',', expand=True)

b = La_Peruse_data[1].str.split(',', expand=True)

df_La_Peruse = pd.concat([a, b], axis=1)

df_cut = df_La_Peruse.drop(df_La_Peruse.columns[14:], axis=1)

columns = ['STN_ID','DATE','TIME','Qflag','LATITUDE','LONGITUDE','DEPTH','VCAR','VTPK','VWH','VCMX','VTP','WDIR','WSPD','WSS','GSPD','WDIR','WSPD','WSS','GSPD','ATMS','ATMS','DRYT','SSTP']

effective = columns[:14]

df_cut.columns = effective
df_cut = df_cut.drop('Qflag',axis =1)

df_cut['Date_time']= df_cut['DATE'] + " " + df_cut['TIME']

LA_PERUSE_array = df_cut.as_matrix()

start_index_2013_Feb  = np.transpose(np.array(np.where(LA_PERUSE_array[:,1] == '02/28/2013')))[0,0]

end_index_2013_September  = np.transpose(np.array(np.where(LA_PERUSE_array[:,1] == '08/31/2013')))[-1,0]

wind_spd = LA_PERUSE_array[start_index_2013_Feb:end_index_2013_September,-2].astype(np.float)

wind_dir = LA_PERUSE_array[start_index_2013_Feb:end_index_2013_September,-3].astype(np.float)

u_wind = np.multiply(wind_spd[:],np.cos(wind_dir))

v_wind = np.multiply(wind_spd[:],np.sin(wind_dir))

time_2013 = LA_PERUSE_array[start_index_2013_Feb:end_index_2013_September,-1].astype(str)

datetime_2013 = np.empty_like(time_2013)

for i in np.arange(datetime_2013.shape[0]):
    datetime_2013[i] = datetime.strptime(time_2013[i], '%m/%d/%Y %H:%M')
    
datetime_2013 = datetime_2013.astype(np.datetime64)

ws_newport = df_subset['WindSpeed']
wd_newport = df_subset['WindDirection(deg)']

ws_La_Perouse = wind_spd
wd_La_Perouse = wind_dir

theta = find_principal_axis_of_dataset_in_degrees(u_data=u_wind, v_data=v_wind)

x_v1, y_v1, x_v2, y_v2 = find_major_eigenvectors(u_data=u_wind, v_data=v_wind)


x_v1_new, y_v1_new, x_v2_new, y_v2_new = find_major_eigenvectors(u_data=u_wind_newport, v_data=v_wind_newport)

ws_newport = df_subset['WindSpeed']
wd_newport = df_subset['WindDirection(deg)']

ws_La_Perouse = wind_spd
wd_La_Perouse = wind_dir

theta = find_principal_axis_of_dataset_in_degrees(u_data=u_wind, v_data=v_wind)


wind_spd = LA_PERUSE_array[start_index_2013_Feb:end_index_2013_September,-2].astype(np.float)

wind_dir = LA_PERUSE_array[start_index_2013_Feb:end_index_2013_September,-3].astype(np.float)

v_wind_rot = np.empty_like(wind_dir)

v_wind_rot[:] = -1*np.multiply(wind_spd[:],np.cos(np.deg2rad(wind_dir[:] - theta)))

# v_wind_rot = np.multiply(wind_spd[:],np.sin(np.deg2rad(wind_dir[:] - 57)))

time_2013 = LA_PERUSE_array[start_index_2013_Feb:end_index_2013_September,-1].astype(str)

datetime_2013 = np.empty_like(time_2013)

for i in np.arange(datetime_2013.shape[0]):
    datetime_2013[i] = datetime.strptime(time_2013[i], '%m/%d/%Y %H:%M')
    
datetime_2013 = datetime_2013.astype(np.datetime64)

range = pd.date_range('2013-02-28', '2013-08-31', freq='D')



La_perouse_series = pd.Series(v_wind_rot, index=datetime_2013)
daily_north_La_Perouse = La_perouse_series.resample('D').mean()

La_perouse_stress_daily = np.sign(daily_north_La_Perouse)*stress(sp=daily_north_La_Perouse)

range = pd.date_range('2013-02-28', '2013-08-31', freq='D')



Newport_series_daily = pd.Series(v_wind_newport_daily)
Newport_stress_daily = np.sign(-Newport_series_daily)*stress(sp=Newport_series_daily)

date1_Newport = '2013-02-28'
date2_Newport = '2013-08-31'
mydates_Newport = pd.date_range(date1_Newport, date2_Newport, freq= 'D')

colors = cm.copper(np.linspace(0, 1, 50))
colors_speed = cm.ocean(np.linspace(0, 1, 50))

mat_file = '/data/ssahu/Falkor_2013/winds/wind_Newport_2013_v6.mat'

mat = scipy.io.loadmat(mat_file)

wind_stress_newport = mat['tau'][:,0]



date1_Newport = '2013-02-28'
date2_Newport = '2013-08-31'
mydates_Newport_1 = pd.date_range(date1_Newport, date2_Newport, freq= 'D')

date1 = '2013-04-29'
date2 = '2013-10-25'
mydates = pd.date_range(date1, date2, freq= 'D')

A1_data = pd.read_table('/data/ssahu/IOS_data/ADCP_E1_A1/a1_20130507_20140512_0456m.csv',delim_whitespace=1,header = None, skiprows=438)

str_stuff = np.array(['Record_Number', 'Date', 'Time', 'Pitch', 'Roll', 'Heading', 'Pressure', 'Temperature'], dtype = 'str')

bin_depths = np.array([80.13,
         96.13,
        112.13,
        128.13,
        144.13,
        160.13,
        176.13,
        192.13,
        208.13,
        224.13,
        240.13,
        256.13,
        272.13,
        288.13,
        304.13,
        320.13,
        336.13,
        352.13,
        368.13,
        384.13,
        400.13,
        416.13,
        432.13], dtype = 'str')

variables = np.array(['vel_north', 'vel_east', 'vel_vert', 'back_scatter_mean'], dtype = 'str')

column_data = []

for j in bin_depths:
    for i in variables:
        column_data = np.append(arr=column_data, values=i+'_'+j)
    

columns_A1 = np.concatenate((str_stuff, column_data), axis = 0)

A1_data.columns = columns_A1

vel_north_80 = np.array(A1_data['vel_north_80.13'], dtype = np.float)[0:5608]
vel_east_80  = np.array(A1_data['vel_east_80.13'], dtype = np.float)[0:5608]
mag_80       = np.sqrt(vel_east_80**2+vel_north_80**2)
mag_80[mag_80 > 2] = 'Nan'

vel_north_96= np.array(A1_data['vel_north_96.13'], dtype = np.float)[0:5608]
vel_east_96 = np.array(A1_data['vel_east_96.13'], dtype = np.float)[0:5608]
mag_96       = np.sqrt(vel_east_96**2+vel_north_96**2)
mag_96[mag_96 > 2] = 'Nan'

vel_north_112 = np.array(A1_data['vel_north_112.13'], dtype = np.float)[0:5608]
vel_east_112  = np.array(A1_data['vel_east_112.13'], dtype = np.float)[0:5608]
mag_112       = np.sqrt(vel_east_112**2+vel_north_112**2)
mag_112[mag_112 > 2] = 'Nan'

vel_north_128 = np.array(A1_data['vel_north_128.13'], dtype = np.float)[0:5608]
vel_east_128  = np.array(A1_data['vel_east_128.13'], dtype = np.float)[0:5608]
mag_128       = np.sqrt(vel_east_128**2+vel_north_128**2)
mag_128[mag_128 > 2] = 'Nan'

vel_north_144 = np.array(A1_data['vel_north_144.13'], dtype = np.float)[0:5608]
vel_east_144  = np.array(A1_data['vel_east_144.13'], dtype = np.float)[0:5608]
mag_144       = np.sqrt(vel_east_144**2+vel_north_144**2)

vel_north_160= np.array(A1_data['vel_north_160.13'], dtype = np.float)[0:5608]
vel_east_160 = np.array(A1_data['vel_east_160.13'], dtype = np.float)[0:5608]
mag_160       = np.sqrt(vel_east_160**2+vel_north_160**2)
mag_160[mag_160 > 2] = 'Nan'

vel_north_176= np.array(A1_data['vel_north_176.13'], dtype = np.float)[0:5608]
vel_east_176 = np.array(A1_data['vel_east_176.13'], dtype = np.float)[0:5608]
mag_176       = np.sqrt(vel_east_176**2+vel_north_176**2)
mag_176[mag_176 > 2] = 'Nan'

vel_north_192 = np.array(A1_data['vel_north_192.13'], dtype = np.float)[0:5608]
vel_east_192  = np.array(A1_data['vel_east_192.13'], dtype = np.float)[0:5608]
mag_192      = np.sqrt(vel_east_192**2+vel_north_192**2)

vel_north_208 = np.array(A1_data['vel_north_208.13'], dtype = np.float)[0:5608]
vel_east_208  = np.array(A1_data['vel_east_208.13'], dtype = np.float)[0:5608]
mag_208      = np.sqrt(vel_east_208**2+vel_north_208**2)

vel_north_224 = np.array(A1_data['vel_north_224.13'], dtype = np.float)[0:5608]
vel_east_224 = np.array(A1_data['vel_east_224.13'], dtype = np.float)[0:5608]
mag_224      = np.sqrt(vel_east_224**2+vel_north_224**2)

vel_north_240 = np.array(A1_data['vel_north_240.13'], dtype = np.float)[0:5608]
vel_east_240 = np.array(A1_data['vel_east_240.13'], dtype = np.float)[0:5608]
mag_240      = np.sqrt(vel_east_240**2+vel_north_240**2)

vel_north_256 = np.array(A1_data['vel_north_256.13'], dtype = np.float)[0:5608]
vel_east_256 = np.array(A1_data['vel_east_256.13'], dtype = np.float)[0:5608]
mag_256      = np.sqrt(vel_east_256**2+vel_north_256**2)

vel_north_272 = np.array(A1_data['vel_north_272.13'], dtype = np.float)[0:5608]
vel_east_272 = np.array(A1_data['vel_east_272.13'], dtype = np.float)[0:5608]
mag_272      = np.sqrt(vel_east_272**2+vel_north_272**2)

vel_north_288 = np.array(A1_data['vel_north_288.13'], dtype = np.float)[0:5608]
vel_east_288 = np.array(A1_data['vel_east_288.13'], dtype = np.float)[0:5608]
mag_288      = np.sqrt(vel_east_288**2+vel_north_288**2)

vel_north_304 = np.array(A1_data['vel_north_304.13'], dtype = np.float)[0:5608]
vel_east_304 = np.array(A1_data['vel_east_304.13'], dtype = np.float)[0:5608]
mag_304      = np.sqrt(vel_east_304**2+vel_north_304**2)

vel_north_320 = np.array(A1_data['vel_north_320.13'], dtype = np.float)[0:5608]
vel_east_320 = np.array(A1_data['vel_east_320.13'], dtype = np.float)[0:5608]
mag_320      = np.sqrt(vel_east_320**2+vel_north_320**2)

vel_north_336 = np.array(A1_data['vel_north_336.13'], dtype = np.float)[0:5608]
vel_east_336 = np.array(A1_data['vel_east_336.13'], dtype = np.float)[0:5608]
mag_336      = np.sqrt(vel_east_336**2+vel_north_336**2)

vel_north_352 = np.array(A1_data['vel_north_352.13'], dtype = np.float)[0:5608]
vel_east_352 = np.array(A1_data['vel_east_352.13'], dtype = np.float)[0:5608]
mag_352      = np.sqrt(vel_east_352**2+vel_north_352**2)


vel_north_368 = np.array(A1_data['vel_north_368.13'], dtype = np.float)[0:5608]
vel_east_368 = np.array(A1_data['vel_east_368.13'], dtype = np.float)[0:5608]
mag_368      = np.sqrt(vel_east_368**2+vel_north_368**2)

vel_north_384 = np.array(A1_data['vel_north_384.13'], dtype = np.float)[0:5608]
vel_east_384 = np.array(A1_data['vel_east_384.13'], dtype = np.float)[0:5608]
mag_384      = np.sqrt(vel_east_384**2+vel_north_384**2)


vel_north_400 = np.array(A1_data['vel_north_400.13'], dtype = np.float)[0:5608]
vel_east_400 = np.array(A1_data['vel_east_400.13'], dtype = np.float)[0:5608]
mag_400      = np.sqrt(vel_east_400**2+vel_north_400**2)

vel_north_416 = np.array(A1_data['vel_north_416.13'], dtype = np.float)[0:5608]
vel_east_416 = np.array(A1_data['vel_east_416.13'], dtype = np.float)[0:5608]
mag_416      = np.sqrt(vel_east_416**2+vel_north_416**2)

vel_north_432 = np.array(A1_data['vel_north_432.13'], dtype = np.float)[0:5608]
vel_east_432 = np.array(A1_data['vel_east_432.13'], dtype = np.float)[0:5608]
mag_432      = np.sqrt(vel_east_432**2+vel_north_432**2)

tarikh = np.array(A1_data['Date'], dtype = np.str)
samai  = np.array(A1_data['Time'], dtype = np.str)

time = np.empty_like(tarikh)

for i in np.arange(tarikh.shape[0]):
    time[i] = tarikh[i] + '' + samai[i]


datetime_A1 = np.empty_like(time)

for i in np.arange(datetime_A1.shape[0]):
    datetime_A1[i] = datetime.strptime(time[i], '%Y/%m/%d%H:%M:%S')
    
datetime_A1 = datetime_A1.astype(np.datetime64)

mydates_A1 = np.array(pd.date_range(datetime_A1[0:5608][0], datetime_A1[0:5608][-1], freq="30min"))

A1_series_east = pd.Series(vel_east_80[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_80_daily_east = daily_east.values
east_80 = np.divide(vel_80_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_80[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_80_daily_north = daily_north.values
north_80 = np.divide(vel_80_daily_north[:], 24)


Speed_80 = np.sqrt(east_80**2 + north_80**2)

A1_series_east = pd.Series(vel_east_96[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_96_daily_east = daily_east.values
east_96 = np.divide(vel_96_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_96[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_96_daily_north = daily_north.values
north_96 = np.divide(vel_96_daily_north[:], 24)


Speed_96 = np.sqrt(east_96**2 + north_96**2)

A1_series_east = pd.Series(vel_east_112[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_112_daily_east = daily_east.values
east_112 = np.divide(vel_112_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_112[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_112_daily_north = daily_north.values
north_112 = np.divide(vel_112_daily_north[:], 24)


Speed_112 = np.sqrt(east_112**2 + north_112**2)

A1_series_east = pd.Series(vel_east_128[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_128_daily_east = daily_east.values
east_128 = np.divide(vel_128_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_128[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_128_daily_north = daily_north.values
north_128 = np.divide(vel_128_daily_north[:], 24)


Speed_128 = np.sqrt(east_128**2 + north_128**2)


A1_series_east = pd.Series(vel_east_144[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_144_daily_east = daily_east.values
east_144 = np.divide(vel_144_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_144[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_144_daily_north = daily_north.values
north_144 = np.divide(vel_144_daily_north[:], 24)


Speed_144 = np.sqrt(east_144**2 + north_144**2)


A1_series_east = pd.Series(vel_east_160[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_160_daily_east = daily_east.values
east_160 = np.divide(vel_160_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_160[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_160_daily_north = daily_north.values
north_160 = np.divide(vel_160_daily_north[:], 24)


Speed_160 = np.sqrt(east_160**2 + north_160**2)

A1_series_east = pd.Series(vel_east_176[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_176_daily_east = daily_east.values
east_176 = np.divide(vel_176_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_176[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_176_daily_north = daily_north.values
north_176 = np.divide(vel_176_daily_north[:], 24)


Speed_176 = np.sqrt(east_176**2 + north_176**2)


A1_series_east = pd.Series(vel_east_192[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_192_daily_east = daily_east.values
east_192 = np.divide(vel_192_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_192[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_192_daily_north = daily_north.values
north_192 = np.divide(vel_192_daily_north[:], 24)


Speed_192 = np.sqrt(east_192**2 + north_192**2)


A1_series_east = pd.Series(vel_east_208[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_208_daily_east = daily_east.values
east_208 = np.divide(vel_208_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_208[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_208_daily_north = daily_north.values
north_208 = np.divide(vel_208_daily_north[:], 24)


Speed_208 = np.sqrt(east_208**2 + north_208**2)


A1_series_east = pd.Series(vel_east_224[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_224_daily_east = daily_east.values
east_224 = np.divide(vel_224_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_224[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_224_daily_north = daily_north.values
north_224 = np.divide(vel_224_daily_north[:], 24)


Speed_224 = np.sqrt(east_224**2 + north_224**2)


A1_series_east = pd.Series(vel_east_240[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_240_daily_east = daily_east.values
east_240 = np.divide(vel_240_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_240[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_240_daily_north = daily_north.values
north_240 = np.divide(vel_240_daily_north[:], 24)


Speed_240 = np.sqrt(east_240**2 + north_240**2)


A1_series_east = pd.Series(vel_east_256[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_256_daily_east = daily_east.values
east_256 = np.divide(vel_256_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_256[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_256_daily_north = daily_north.values
north_256 = np.divide(vel_256_daily_north[:], 24)


Speed_256 = np.sqrt(east_256**2 + north_256**2)


A1_series_east = pd.Series(vel_east_272[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_272_daily_east = daily_east.values
east_272 = np.divide(vel_272_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_272[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_272_daily_north = daily_north.values
north_272 = np.divide(vel_272_daily_north[:], 24)


Speed_272 = np.sqrt(east_272**2 + north_272**2)


A1_series_east = pd.Series(vel_east_288[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_288_daily_east = daily_east.values
east_288 = np.divide(vel_288_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_288[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_288_daily_north = daily_north.values
north_288 = np.divide(vel_288_daily_north[:], 24)


Speed_288 = np.sqrt(east_288**2 + north_288**2)


# A1_series_east = pd.Series(vel_east_304[1200:2560], index=mydates_A1)
# daily_east = A1_series_east.resample('1440T').sum()
# vel_304_daily_east = daily_east.values
# east_304 = np.divide(vel_304_daily_east[:], 24)

# A1_series_north = pd.Series(vel_north_304[1200:2560], index=mydates_A1)
# daily_north = A1_series_north.resample('1440T').sum()
# vel_304_daily_north = daily_north.values
# north_304 = np.divide(vel_304_daily_north[:], 24)


# Speed_304 = np.sqrt(east_304**2 + north_304**2)

A1_series_east = pd.Series(vel_east_320[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_320_daily_east = daily_east.values
east_320 = np.divide(vel_320_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_320[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_320_daily_north = daily_north.values
north_320 = np.divide(vel_320_daily_north[:], 24)


Speed_320 = np.sqrt(east_320**2 + north_320**2)


A1_series_east = pd.Series(vel_east_336[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_336_daily_east = daily_east.values
east_336 = np.divide(vel_336_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_336[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_336_daily_north = daily_north.values
north_336 = np.divide(vel_336_daily_north[:], 24)


Speed_336 = np.sqrt(east_336**2 + north_336**2)

A1_series_east = pd.Series(vel_east_352[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_352_daily_east = daily_east.values
east_352 = np.divide(vel_352_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_352[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_352_daily_north = daily_north.values
north_352 = np.divide(vel_352_daily_north[:], 24)


Speed_352 = np.sqrt(east_352**2 + north_352**2)

# A1_series_east = pd.Series(vel_east_368[1200:2560], index=mydates_A1)
# daily_east = A1_series_east.resample('1440T').sum()
# vel_368_daily_east = daily_east.values
# east_368 = np.divide(vel_368_daily_east[:], 24)

# A1_series_north = pd.Series(vel_north_368[1200:2560], index=mydates_A1)
# daily_north = A1_series_north.resample('1440T').sum()
# vel_368_daily_north = daily_north.values
# north_368 = np.divide(vel_368_daily_north[:], 24)


# Speed_368 = np.sqrt(east_368**2 + north_368**2)


A1_series_east = pd.Series(vel_east_384[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_384_daily_east = daily_east.values
east_384 = np.divide(vel_384_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_384[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_384_daily_north = daily_north.values
north_384 = np.divide(vel_384_daily_north[:], 24)


Speed_384 = np.sqrt(east_384**2 + north_384**2)


A1_series_east = pd.Series(vel_east_368[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_368_daily_east = daily_east.values
east_368 = np.divide(vel_368_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_368[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_368_daily_north = daily_north.values
north_368 = np.divide(vel_368_daily_north[:], 24)


Speed_368 = np.sqrt(east_368**2 + north_368**2)


A1_series_east = pd.Series(vel_east_304[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_304_daily_east = daily_east.values
east_304 = np.divide(vel_304_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_304[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_304_daily_north = daily_north.values
north_304 = np.divide(vel_304_daily_north[:], 24)

Speed_304 = np.sqrt(east_304**2 + north_304**2)



A1_series_east = pd.Series(vel_east_400[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_400_daily_east = daily_east.values
east_400 = np.divide(vel_400_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_400[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_400_daily_north = daily_north.values
north_400 = np.divide(vel_400_daily_north[:], 24)


Speed_400 = np.sqrt(east_400**2 + north_400**2)

A1_series_east = pd.Series(vel_east_416[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_416_daily_east = daily_east.values
east_416 = np.divide(vel_416_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_416[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_416_daily_north = daily_north.values
north_416 = np.divide(vel_416_daily_north[:], 24)


Speed_416 = np.sqrt(east_416**2 + north_416**2)


A1_series_east = pd.Series(vel_east_432[:], index=mydates_A1)
daily_east = A1_series_east.resample('1440T').sum()
vel_432_daily_east = daily_east.values
east_432 = np.divide(vel_432_daily_east[:], 24)

A1_series_north = pd.Series(vel_north_432[:], index=mydates_A1)
daily_north = A1_series_north.resample('1440T').sum()
vel_432_daily_north = daily_north.values
north_432 = np.divide(vel_432_daily_north[:], 24)


Speed_432 = np.sqrt(east_432**2 + north_432**2)

date1 = '2013-05-07'
date2 = '2013-08-31'
mydates_A1_1 = pd.date_range(date1, date2, freq= 'D')

arrays_east = [east_80, east_96, east_112, east_128, east_144, east_160, east_176, east_192, east_208,\
               east_224, east_240, east_256, east_272, east_288, east_304, \
               east_320, east_336, east_352, east_368, east_384, east_400, east_416, \
              east_432]
arrays_north = [north_80, north_96, north_112, north_128, north_144, north_160, north_176, north_192, \
                north_208, north_224, north_240, north_256, north_272, north_288, \
                north_304, north_320, north_336, north_352, north_368, north_384, \
               north_400, north_416, north_432]
arrays_speed = [Speed_80, Speed_96, Speed_112, Speed_128, Speed_144, Speed_160, Speed_176, Speed_192, \
                Speed_208, Speed_224, Speed_240, Speed_256, Speed_272, Speed_288, \
                Speed_304, Speed_320, Speed_336, Speed_352, Speed_368, Speed_384, \
               Speed_400, Speed_416, Speed_432]


z_ADCP = np.array(bin_depths.astype(float))

east_ADCP_A_1_august = np.stack(arrays=arrays_east, axis=1)
north_ADCP_A_1_august = np.stack(arrays=arrays_north, axis=1)
speed_ADCP_A_1_august = np.stack(arrays=arrays_speed, axis=1)


zlevels = nc.Dataset('/data/mdunphy/NEP036-N30-OUT/CDF_COMB_COMPRESSED/NEP036-N30_IN_20140915_00001440_grid_T.nc').variables['deptht'][:32]
y_wcvi_slice = np.array(np.arange(180,350))
x_wcvi_slice = np.array(np.arange(480,650))

u_vel = np.empty((240,zlevels.shape[0],y_wcvi_slice.shape[0]+1,x_wcvi_slice.shape[0]+1))
v_vel = np.empty((240,zlevels.shape[0],y_wcvi_slice.shape[0]+1,x_wcvi_slice.shape[0]+1))



i = 0

for file in sorted(glob.glob('/data/ssahu/NEP36_2013_summer_hindcast/two_months_more_velocities_total_for_a_notebook/\
cut_NEP36-S29_1d*grid_U*.nc')):
    vector_u = U_timeseries_at_WCVI_locations(nc.Dataset(file))
    u_vel[i:i+10,...] = vector_u[0]
    i = i+10

j = 0
for file in sorted(glob.glob('/data/ssahu/NEP36_2013_summer_hindcast/two_months_more_velocities_total_for_a_notebook/\
cut_NEP36-S29_1d*grid_V*.nc')):
    vector_v = V_timeseries_at_WCVI_locations(nc.Dataset(file))
    v_vel[j:j+10,...] = vector_v[0]
    j = j+10
    
    
u_tzyx, v_tzyx = viz_tools.unstagger(u_vel, v_vel)

mag_vel = np.sqrt(np.multiply(u_tzyx,u_tzyx) +  np.multiply(v_tzyx,v_tzyx));
ang_vel = np.degrees(np.arctan2(v_tzyx, u_tzyx))

u_unrotated = u_tzyx
v_unrotated = v_tzyx


file_model = nc.Dataset('/data/ssahu/NEP36_2013_summer_hindcast/cut_NEP36-S29_1d_20130429_20131025_grid_T_20130429-20130508.nc')

lon = file_model.variables['nav_lon'][1:,1:]
lat = file_model.variables['nav_lat'][1:,1:]


lon_A1 = -126.20433
lat_A1 = 48.52958

j, i = geo_tools.find_closest_model_point(lon_A1,lat_A1,\
                                          lon,lat,grid='NEMO',tols=\
                                          {'NEMO': {'tol_lon': 0.1, 'tol_lat': 0.1},\
                                           'GEM2.5': {'tol_lon': 0.1, 'tol_lat': 0.1}}) 

date1 = '2013-02-28'
date2 = '2013-08-31'
mydates_A1_model = pd.date_range(date1, date2, freq= 'D')

along_model =np.empty_like(v_unrotated[0:-55,:,j,i])

for k in np.arange(1,along_model.shape[1]):
    along_model[:,k] = get_alongshore_currents(u_curr=u_unrotated[:-55,k,j,i], \
                                                       v_curr= v_unrotated[:-55,k,j,i])
    
along_ADCP_A1_august = np.empty_like(north_ADCP_A_1_august)

for k in np.arange(1,along_ADCP_A1_august.shape[1]):
    along_ADCP_A1_august[:,k] = get_alongshore_currents(u_curr=east_ADCP_A_1_august[:,k], \
                                                       v_curr= north_ADCP_A_1_august[:,k])
north_80_rot = along_ADCP_A1_august[:,0] 

along_ADCP_A1_august[along_ADCP_A1_august<-2] = 'Nan'
along_ADCP_A1_august[along_ADCP_A1_august>2] = 'Nan'

nans = np.where(np.isnan(along_ADCP_A1_august))

for t in np.arange(along_ADCP_A1_august.shape[0]):
    
    nans, adcp_data_x = nan_helper(along_ADCP_A1_august[t,:])
    
    along_ADCP_A1_august[t,nans]= np.interp(adcp_data_x(nans), adcp_data_x(~nans), along_ADCP_A1_august[t,~nans])
    
along_ADCP_A1_august[:,0] = north_80_rot

streng = np.empty((along_ADCP_A1_august.shape[0]))

for t in np.arange(along_ADCP_A1_august.shape[0]):
    streng[t] = calculate_strength_of_adcp_undercurrent(t, adcp_alongshore=along_ADCP_A1_august[t,:],\
                                                          adcp_depths=z_ADCP)


streng_model = np.empty((along_model.shape[0]))

for t in np.arange(along_model.shape[0]):
    streng_model[t] = calculate_strength_of_model_undercurrent(t,model_alongshore=along_model[t,:],model_depths=zlevels)
    

step = 1
window_size_1 = 6
time_axis = datetime_2013[::step]


s_v_newport = v_wind_newport[:]   
filt_v_newport = smooth(s_v_newport,window_size_1,'hanning')
Newport_stress = np.sign(filt_v_newport)*stress(sp=filt_v_newport)

s_v = La_perouse_stress_daily  
filt_v_La_Perouse = smooth(s_v,window_size_1,'hanning')

s_v = wind_stress_newport[58:243] 
filt_v_newport = smooth(s_v,window_size_1,'hanning')


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

sns.set_context('talk')

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
        
    norm = mpl.colors.Normalize(vmin=-0.2,vmax=0)


    plt.rcParams['contour.negative_linestyle'] = 'solid' # default is to have negative contours with dashed lines
    plt.rcParams.update({'font.size':16})




    cmap = plt.get_cmap(cmo.cm.balance)
#     cmap = plt.get_cmap('vlag')
    cmap.set_bad('#8b7765')

    V_normalized = (spic_iso - spic_iso.min().min())
    V_normalized = V_normalized / V_normalized.max().max()
    spic_plot = V_normalized


    ls = LightSource(290, 35)
    #     cmap1 = ls.shade(spic_iso, cmap=cmap, vert_exag=0.8, vmin=-0.4, vmax =-0.15, blend_mode='overlay')

    cmap1 = ls.shade(spic_iso, cmap=cmap, vert_exag=0.8, vmin=-0.2, vmax =0, blend_mode='overlay')




    fig = plt.figure(figsize=(15, 25))
    
    ax2 = fig.add_subplot(613)
#     gridspec = {'height_ratios': [1, 1]}

#     fig, (ax11, ax2) = plt.subplots(2, 1, figsize=(25, 20), gridspec_kw=gridspec)
    
        #For the temporal correlation plot
    
    
    ax2.plot(mydates_Newport , filt_v_La_Perouse, color = colors[15],  linewidth=2, label = 'La Perouse')
    ax2.plot(mydates_Newport_1, filt_v_newport, c = colors[30], linewidth=2, label = 'Newport')

    # ax2.plot(mydates_Newport , La_perouse_stress_daily, color = colors[15],  linewidth=2, label = 'La Perouse')

    # ax2.plot(mydates_Newport_1, wind_stress_newport[58:243], c = colors[30], linewidth=2, label = 'Newport')


    ax2.set_ylabel('Alongshore Wind Stress ($\mathrm{N/m^2}$)', fontsize = 16)
    ax2.set_ylim(-0.15, 0.15)
    ax2.tick_params(axis='both',labelsize =16, color = colors[25])
    ax2.yaxis.label.set_color(colors[25])
    ax2.legend(loc = 'upper left', fontsize  =14)
    ax2.spines['left'].set_color(colors_speed[25])
    # ax2.axhline(y=0, color = 'k')
    ax2.axvline(x = mydates_Newport[mydates_Newport == '2013-08-21'], color = 'r', linestyle  = '--')
    if (t < 125):
        ax2.axvline(x = mydates_Newport[t+60], color = 'g', linestyle  = '--')
    ax2.axvline(x = mydates_Newport[-1], color = 'r', linestyle  = '--')

#     fig.autofmt_xdate()


    ax1 = ax2.twinx()

    ax1.plot(mydates_A1_1, streng, color = colors_speed[20], linewidth=1.5, linestyle = 'dashed', label = 'ADCP at A1')
    ax1.plot(mydates_A1_model, streng_model, color = colors_speed[20], linewidth=2, label = 'NEP36 at A1')

    ax1.set_ylabel('Undercurrent Transport ($\mathrm{m^2/s}$)', fontsize = 16)
    ax1.set_ylim(-2, 63)
    ax1.tick_params(axis='both',labelsize =16, color = colors_speed[25])
    ax1.yaxis.label.set_color(colors_speed[25])
    ax1.legend(loc = 'upper right', fontsize  =14)
    ax1.spines['right'].set_color(colors_speed[25])

    ax2.grid()
    
#     ax = fig.gca(projection='3d')
    ax = fig.add_subplot(212, projection='3d')
    
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
        month = 'July'
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
    surf = ax.plot_surface(lon_wcvi, lat_wcvi, -depth_rho_0[:,:], facecolors=cmap1, linewidth=0, antialiased=False, rstride=1, cstride=1)
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
    ax.set_ylabel('Latitude (N)', fontsize = 18, labelpad= 18)
    ax.set_xlabel('Longitude (E)', fontsize = 18, labelpad= 18)
    ax.set_zlabel('Depth (m)', fontsize = 18, labelpad= 18)
    ax.set_zlim(-250,0)
    m = cm.ScalarMappable(cmap=plt.get_cmap(cmo.cm.balance))
    m.set_array(spic_iso)
    #     m.set_clim(-0.4, -0.05)
    m.set_clim(-0.2, 0)
    plt.colorbar(m)
    ax.set_aspect('auto')
    ax.legend(loc='best', fancybox=True, framealpha=0.25)
    ax.view_init(40, 240) # elevation and azimuth
    
    fig.tight_layout()
            
    plt.savefig('/home/ssahu/saurav/3D_images_for_video_spice/2013_rho_26_4_{0}.png'.format(t))
    plt.close()


rho_0 = 26.4

print("Plotting Begins")

for t in np.arange(rho.shape[0]):
    print (t)
    plot_iso_den(t, rho_0)
    
print("The code has run to completion, thanks for waiting")
    
