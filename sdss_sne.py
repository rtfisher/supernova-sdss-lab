###!/usr/bin/python

#################################################################
##
## sdss_sne 
##
## Module to read in SNe Ia light curve data from SDSS files.
## Updated to Python 3.x.
##
## Contains functions
##
##  readsn : Usage   snstring, dates, mags = readsn (file, verbose = 1)
##
##  plotsn : Usage   plotsn (file)
##
##  saveplotsn : Usage    saveplotsn (file)
##
## Last udpated: rtf 10/9/2017
## 
#################################################################

import string
import numpy as np
import matplotlib.pyplot as plt

def readsn (file, verbose = 1):
# Verbose flag ( = 1 for verbose output, 0 if not)

# Open data file for reading

   f = open (file, 'r')

# Inititalize data lists

   dates = []
   magnitudes = []
   magnitudeserror = []

# Read data in line by line

# First line of header

   line = f.readline ()
   if (verbose == 1) :
      print ()
      print ("Header information")
      print ("==================")
      print (line)

# Split first line into strings

   lst = line.split ()

# Grab the name of the SN from first line

   snstring = lst [5];

# Second line of header - grab redshift and redshifterr

   line = f.readline ()
   if (verbose == 1) :
      print (line)
   lst = line.split ()
   redshift = float (lst [2]);
   redshifterr = float (lst [4]);

# Next lines in header

   for line in f:
      if line.startswith("#") :   # Comment fields begin with number sign
         if (verbose == 1) :
            print (line)
      else :

# Split the line of data into fields

            lst = line.split()

#  FLAG: gives values of the photometry flags (see paper) for each measurement.
#   FLAG==0 means there are no liens on the measurement.
#   FLAG>1024 means the measurement is likely bad. For 0<FLAG<1024, the
#   measurement is likely OK, but there was some lien on the frames so it was
#   not used to constrain the galaxy background (except for 16 and 32).
#   See paper for more details.

            flag = float (lst [0])

# Modified Julian date of measurement

            mjd = float (lst [1])

# Filter in which measurement was made

            filter = int (lst [2])

# SDSS asinh magnitude

            mag = float (lst [3])

# Error on magnitude

            magerr = float (lst [4])

# Estimate of systematic error from uncertainty in sky determination

            skyerr = float (lst [5])

# Estimate of systematic error from uncertainty in galaxy background determination

            galaxyerr = float (lst [6])

# Derived flux in microJansky

            flux = float (lst [7])

# Error in the flux

            fluxerr = float (lst [8])

# Sky error in flux units

            skyerrflux = float (lst [9])

# Galaxy error in flux units

            galaxyerrflux = float (lst [10])

# If g band filter

            if (filter == 1) :
               dates.append (mjd)
               magnitudes.append (mag)
               magnitudeserror.append (magerr)

            if (verbose == 1) :
               print (flag, ' ', mjd, ' ', filter, ' ', mag, ' ', magerr, ' ', skyerr, ' ', galaxyerr, ' ', flux, ' ', fluxerr)

   print ()
   print ("Dates = ", dates)
   print ('Analyzing supernova ', snstring)
   print ('redshift = ', redshift, '+/-', redshifterr)

   return snstring, dates, magnitudes, magnitudeserror


def plotsn (file):

   snstring, dates, mags, magerr = readsn (file)
   plt.errorbar (dates, mags, yerr=magerr, fmt = 'ro', linestyle = '-')
   plt.plot (dates, mags)
   ax = plt.gca()
   ax.set_ylim(ax.get_ylim()[::-1])
   ax.set_xlabel ("Time (MJD)")
   ax.set_ylabel ("Magnitude")
   ax.set_title ("Light Curve SN " + snstring)
   plt.show ()


def saveplotsn (file):

   snstring, dates, mags, magerr = readsn (file)
   plt.errorbar (dates, mags, yerr=magerr, fmt = 'ro', linestyle = '-')
   plt.plot (dates, mags)
   ax = plt.gca()
   ax.set_ylim(ax.get_ylim()[::-1])
   ax.set_xlabel ("Time (MJD)")
   ax.set_ylabel ("Magnitude")
   ax.set_title ("Light Curve SN " + snstring)
   plt.savefig (snstring + '_magn', format = 'pdf')
