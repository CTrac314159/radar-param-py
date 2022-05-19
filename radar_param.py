# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 13:13:51 2020

@author: Chris Tracy

This module contains functions for computing several different radar parameters from
CSV drop size distribution (DSD) data from a 2DVD (disdrometer) instrument. Plots are also
made in each of the functions.

Routine Listing
-------------------------------------

RadarParam Class:
    refl_nd : Calculates the reflectivity factor (z) from the DSD file and plots a time
              series
    refl_npol : Extracts and plots the dual-pol reflectivity data from the appropriate file
    z_r : Estimates the rain rate from the DSD data and derives a z-R relationship from
          the rain rate and calculated reflectivity factor from refl_nd. Plots the z-R
          relationship fit with the "true" scatter points
rel_gain : Calculates and plots the relative gain of a radar beam at a specified beamwidth
beamwidth : Calculates and plots the beamwidth of a radar given a specified wavelength. 
            Can then be used when calling rel_gain.
mdr : Calculates and plots the minimum detectable reflectivity of a radar given a specified 
      constant, minimum received power, and a radar loss term
"""

#Import the global packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates
import pandas as pd

class RadarParam:
    
    def __init__(self, nd_file, npol_file, d_min = 0.3, d_max = 9.9):
        """
        Parameters
        ----------
        nd_file : The CSV file string path with the DSD radar data
        npol_file : The CSV file string path with the NPOL radar data
        d_min : The lower-end drop size (mm), optional float. The default is 0.3 mm.
        d_max : The higher-end drop size (mm), optional float. The default is 9.9 mm.
        """
        
        self.nd = pd.read_csv(nd_file).T
        self.npol = pd.read_csv(npol_file).T
        self.d_min = d_min
        self.d_max = d_max

    def refl_nd(self, plot = True):
        """
        Parameters
        ----------
        d_min : The lower-end drop size (mm), optional float. The default is 0.3 mm.
        d_max : The higher-end drop size (mm), optional float. The default is 9.9 mm.
        plot : Boolean on whether to plot the time series, optional. The default is True. 
    
        Returns
        ----------
        nd : The transposed DSD DataFrame.
        z : The calculated reflectivity factor 2D array, same shape as nd.
        d_bins : The 1D array of size bins, same length as the number of transposed
                 DataFrame columns.
        delta_d : The calculated size interval based on the number of transposed DataFrame
                  columns and the specified min and max drop size.
        """
        
        #Define the size bins and the interval from the specified max and min size along
        #with the number of columns. Make sure that the diameter bins matches the number of columns.
        self.delta_d = (self.d_max - self.d_min)/self.nd.shape[1]
        self.d_bins = np.arange(self.d_min, self.d_max, self.delta_d)
        
        #Calculate the reflectivity factor for all cells, then sum up the values along
        #each row. Convert to the logarithmic scale (dBZ).
        self.z = (self.d_bins**6) * self.nd.values[:] * self.delta_d
        self.z = self.z.sum(axis = 1)
        Z = 10 * np.log10(self.z)
        
        if plot:
            
            #Plot the reflectivity factor on the left subplot and the logarithmic Z (dBZ)
            #on the right subplot. Extract either AM or PM from the transposed CSV index.
            times = pd.to_datetime(self.nd.index, format = '%H:%M %p')
            fig, ax = plt.subplots(ncols = 2, figsize = (12, 4))
            ax[0].plot(times, self.z)
            ax[0].set_title('2DVD Radar Reflectivity Factor (z) Time Series')
            ax[0].set_xlabel('Time Stamp (' + self.nd.index[0][-2:] + ')')
            ax[0].set_ylabel('z' r'$\ (\frac{mm^6}{m^3}$)')
            ax[0].get_xaxis().set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
            ax[1].plot(times, Z)
            ax[1].set_title('2DVD Radar Reflectivity Factor (Z) Time Series')
            ax[1].set_xlabel('Time Stamp (' + self.nd.index[0][-2:] + ')')
            ax[1].set_ylabel('Z (dBZ)')
            ax[1].get_xaxis().set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        
        return self.nd, self.z, self.d_bins, self.delta_d
    
    def refl_npol(self):
        """
        Parameters
        ----------
        file : The string referencing the CSV file with the dual-pol reflectivity data
    
        Returns
        ----------
        npol : The transposed DataFrame of the reflectivity data.
        """
        
        #Read in the DSD file to get "AM" or "PM".
        am_pm = self.nd.index[0][-2:]
        
        #Convert the time index into a Datetime format for the plot.
        times1 = pd.to_datetime(self.npol.index, format = '%H:%M')
        
        #Plot the data. Extract either AM or PM from the transposed CSV index.
        fig, ax = plt.subplots(figsize = (7,5))
        ax.plot(times1, self.npol)
        ax.set_title('NPOL Radar Reflectivity Factor (Z) Time Series')
        ax.set_xlabel('Time Stamp (' + am_pm + ')')
        ax.set_ylabel('Z (dBZ)')
        ax.get_xaxis().set_major_formatter(matplotlib.dates.DateFormatter('%H:%M'))
        
        return self.npol
    
    def z_r(self):
        """
        Parameters
        ----------
        file : The string referencing the CSV file with the DSD data.
        d_min : The lower-end drop size (mm), optional float. The default is 0.3 mm.
        d_max : The higher-end drop size (mm), optional float. The default is 9.9 mm.
    
        Returns
        ----------
        z_r_pred : The predicted reflectivity factor array using the derived z-R relationship.
        R.values : The estimated rain rate array from the DSD data. Should be the same
                   shape as z_r_pred.
        param : The curve fit z-R relationship parameters, B and A respectively in the array.
        cov : The covariance matrix array associated with the derived z-R relationship.
        """
        
        from scipy.optimize import curve_fit
        
        #Next, a relationship for rain rate is developed and plotted. Assume values of
        #alpha and beta below for simplicity.
        a = 3.78
        self.refl_nd(plot = False)
        R = (((3.6 * 10**-3) * np.pi * a)/6.) * self.nd * (self.d_bins**3.67) * self.delta_d
        R = R.sum(axis = 1)
        
        #Get z-R relation from the rain rate data using the specified A and B constant parameters.
        def eq(A, B, R):
            z_r = A * (R**B)
            return z_r
        
        #Make the curve fit using the above function. Predict the reflectivity factor using
        #said function over a reasonable range of rain rates.
        param, cov = curve_fit(eq, R, self.z)
        r_bins = np.linspace(0, 60, R.shape[0])
        z_r_pred = eq(param[1], param[0], r_bins)
        
        #Plot the data
        fig, ax = plt.subplots(figsize = (6, 4))
        plt.scatter(R, self.z, s = 1.3)
        plt.plot(r_bins, self.z_r_pred, color = 'r')
        plt.title('Rain Rate vs. Reflectivity Factor with Curve Fit')
        plt.xlabel('Rain Rate' r'($\ \frac{mm}{h}$)')
        plt.ylabel('Reflectivity Factor z' r'$\ (\frac{mm^6}{m^3}$)')
        plt.text(5, 10**5, 'z = ' + str(round(param[1], 2)) + ' * R$^{{{}}}$'.format(round(param[0], 2)))
        
        return z_r_pred, R.values, param, cov
    
def rel_gain(theta_0, theta = np.arange(-3, 3, 0.5)):
    """
    Parameters
    ----------
    theta_0 : The radar beamwidth in degrees. Can either be an integer or float.
    theta : The range of angular distances to use in degrees, optional. The default 
            is np.arange(-3, 3, 0.5).

    Returns
    ----------
    rel_gain : The calculated relative gain data, same array shape as theta.
    """
    
    #Define a range of angular distances. Calculate the relative gain with these
    #and the beamwidth. Plot the data.
    rel_gain = ((-40. * np.log(2))/np.log(10)) * ((theta/theta_0)**2)
    fig, ax = plt.subplots()
    ax.plot(theta, rel_gain)
    ax.set_title('Relative Gain vs. Angular Distance for ' + str(theta_0) + '$^{\circ}$ Beamwidth')
    ax.set_xlabel('Angular Distance from Main Lobe (degrees)')
    ax.set_ylabel('Relative Gain (dB)')
    
    return rel_gain

def beamwidth(wavelength, diam = np.arange(0.3, 9.0, 0.5)):
    """
    Parameters
    ----------
    wavelength : The wavelength of the radar in cm. Can be either a float or integer.
    diam : The range of reflector diameters to use in meters, optional. The default 
           is np.arange(0.3, 9.0, 0.5).

    Returns
    -------
    beamwidth : The calculated beamwidth data, same array shape as diam.
    """
    
    #Calculate the beamwidth of the radar using the wavelength and reflector diameter.
    #Include conversion factor to convert the wavelength to meters.
    beamwidth = (180./np.pi) * np.sqrt(1.6) * (wavelength/(diam * 100))
    fig, ax = plt.subplots()
    ax.plot(diam, beamwidth)
    ax.set_title('Beamwidth vs. Diameter at ' + str(wavelength) + ' cm wavelength')
    ax.set_xlabel('Diameter of Reflector (meters)')
    ax.set_ylabel('Beamwidth (degrees)')
    
    return beamwidth

def mdr(C, P_r, L, slant_range = np.arange(5, 300, 5)):
    """
    Parameters
    ----------
    C : The radar constant in dB. Can either be a float or integer.
    P_r : The minimum received power of the radar in dBm. Can either be a float or
          integer.
    L : The radar loss term in dB. Can either be a float or integer.
    slant_range : The array of slant ranges to use in km, optional. The default is 
                  np.arange(5, 300, 5).

    Returns
    -------
    mdr : The calculated MDR data, same array shape as slant_range.
    """
    
    #Using the y-intercept and the array of slant ranges, calculate the minimum detectable effective 
    #reflectivity factor and plot the data.
    y_0 = C + P_r - L
    mdr = (20 * np.log10(slant_range)) + y_0
    fig, ax = plt.subplots()
    ax.plot(slant_range, mdr)
    ax.set_title('Minimum Detectable Effective Reflectivity Factor (MDR)')
    ax.set_xlabel('Slant Range (km)')
    ax.set_ylabel('MDR ($dBZ_{e}$)')
    
    return mdr