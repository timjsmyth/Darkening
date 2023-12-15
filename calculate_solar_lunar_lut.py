#!/usr/bin/env python3
import os
import sys
from timeit import default_timer as timer
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from itertools import product
from random import randrange, uniform
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from PIL import Image
from osgeo import gdal
from numpy import zeros, newaxis
from scipy.ndimage import zoom
from matplotlib.colors import LogNorm
from mpl_toolkits.basemap import Basemap
import argparse
import warnings
import pdb
import datetime
from pysolar.solar import get_altitude 
from pysolar.radiation import get_air_mass_ratio
import gcirrad
import SpectralSplit
import spectres
import pandas as pd
import scipy.integrate
import astropy.utils.iers
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_moon, Angle
from astroplan import moon
warnings.simplefilter('ignore', np.RankWarning)

def get_air_mass_kasten_young(altitude_deg):
    # Original C
    # return 1.0 / (sin(altitude_deg * DtoR) + 0.50572 * pow(altitude + 6.07995, -1.6364));
    return 1.0 / (np.sin(np.radians(altitude_deg)) + 0.50572 * (altitude_deg + 6.07995)**(-1.6364))

# Lumme-Bowell (1981) 
# phase_angle in radians 
def lumme_bowell(phase_angle):
    Q = 0.108    # multiple-scattering fraction 

    #TO DO: find more precise angle for smooth curve? 
    if (np.degrees(phase_angle) < 25.07):
       Phi1 = 1.0 - np.sin(phase_angle) / (0.124 + 1.407 * np.sin(phase_angle) - 0.758 * np.sin(phase_angle) * np.sin(phase_angle))
    else:
       Phi1 = np.exp(-3.343 * (np.tan(phase_angle / 2.0))**0.632)

    # multiple scattering 
    Phi_m = 1.0 / np.pi * (np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle))

    phase_factor = (1.0 - Q) * Phi1 + Q * Phi_m

    return phase_factor   
       

def main():

   parser = argparse.ArgumentParser(description='Produce Lunar and Solar monthly surface irradiances')
   year = 2022
   solar_day = 15
   solar_hour = 12
   lunar_hour = 0
   minute = 0
   longitude_deg = 0 # calculation only a function of latitude

   #Moon culmination dates at 50N in 2022 (from mooncalc.org - which unhelpfully adds daylight saving)
   #Jan 18 00:15
   #Feb 15 23:48
   #Mar 18 00:01
   #Apr 17 00:13 GMT
   #May 15 23:46 GMT
   #Jun 13 23:27 GMT
   #Jul 14 00:24 GMT
   #Aug 12 00:08 GMT
   #Sep 09 23:43 GMT
   #Oct 08 23:10 GMT
   #Nov 07 23:20
   #Dec 07 23:41
   full_moon_dates = [18, 15, 18, 17, 15, 13, 14, 12, 9, 8, 7, 7] # Full moon dates in 2022 (year had one full moon in each month)
   full_moon_hours = [0, 23, 0, 0, 23, 23, 0, 0, 23, 23, 23, 23]
   full_moon_minutes = [15, 48, 1, 13, 46, 27, 24, 8, 43, 10, 20, 41]
   month_str = ['01','02','03','04','05','06','07','08','09','10','11','12']

   # 0.000064692 = sr value for subtending angle of the lunar disc (0.26 degree semi-diameter --> sr)
   lun_sva = 0.000064692/np.pi 
   
   ##  Split Spectra of Sun and Moon
   # blockPrint()
   path_name = os.getcwd() + "/Required/" 
   fname_Sol = path_name + 'Solarspectra.csv'
   fname_Lun = path_name + 'Moon_spectra.csv'
   fname_ALAN = path_name + 'Lightspectra.csv'
   fname_gcirrad = path_name + 'gcirrad.dat'
   # Table 2 from Velikodsky et al., (2011) New Earth-based absolute photometry of the Moon, Icarus, 214, 30 - 45
   fname_lunar_albedo = path_name + 'lunar_albedo.dat' 

   df_trans_atmos = pd.read_csv(fname_gcirrad, delim_whitespace=True)
   df_k_atmos = SpectralSplit.k_spectral_split('Wavelength','Trans',df_trans_atmos)
   k_atmos_bb = df_k_atmos['Broadband'].to_numpy()[0]
   df_k_atmos_spec = df_k_atmos[['Red','Green','Blue']]
   
   # Lunar spectral albedo
   df_spectral_lunar_albedo = pd.read_csv(fname_lunar_albedo, delim_whitespace=True)
   df_albedo_lunar = SpectralSplit.lunar_albedo_spectral_split('Wavelength','AverageMoonAlbedo',df_spectral_lunar_albedo)

   # interpolate the lunar spectral albedo onto the atmospheric transmission wavelengths
   gcirrad_wavs = df_trans_atmos['Wavelength'].to_numpy()
   lunar_albedo_wavs = df_spectral_lunar_albedo['Wavelength'].to_numpy()
   spec_albedo = df_spectral_lunar_albedo['AverageMoonAlbedo'].to_numpy()
   lunar_UV_albedo_fill = 0.0728 # for wavelengths shorter than 350 nm (305 - 350 nm)
   lunar_albedo_gcirrad = spectres.spectres(gcirrad_wavs, lunar_albedo_wavs, spec_albedo, fill=lunar_UV_albedo_fill, verbose=False)    
   # add column into the gcirrad dataframe
   df_trans_atmos['LunarAlbedo'] = lunar_albedo_gcirrad

   parser.add_argument("--odir", type=str, default='', help="Output directory for netcdf imagery")
   args = parser.parse_args()

   filestr_box = ""
   cloudy = "clear"

   if args.odir:
      outdir = args.odir 
   else:
      outdir = '/home/scratch/data/workspace/ALAN/Darkening/'
      filedesc = "-Darkening-"      

   # fix for issue with np.bool
   np.bool = np.bool_

   # Read in a template file which contains the lat and lons for 9 km netcdf
   datadir = '/home/scratch/data/workspace/ALAN/Darkening/template'

   for filename in sorted(os.listdir(datadir)):
      if filename.endswith(".nc"): 
         # determine the filename for the netCDF and read in the file
         #print(os.path.join(datadir, filename))
         print("=======================================")
         print(" Input template filename: ", filename)
         fname = os.path.join(datadir, filename)
         infile=nc.Dataset(fname, 'r')
         nc_dims = [dim for dim in infile.dimensions]  # list of nc dimensions
         
         lats=infile.variables['lat'][:]
         lats_shape = lats.shape
         lons=infile.variables['lon'][:]
         Kd_490=infile.variables['Kd_490'][:]
         Kd_490_shape = Kd_490.shape

   for month in range(1,13):
      print("=======================================")
      ofname = "Surface_Irradiance_Solar_Lunar"+filedesc+"clear-"+month_str[month-1]+"-2022-9km.nc"

      solar_490nm = np.zeros(Kd_490.shape)
      solar_blue = np.zeros(Kd_490.shape)
      solar_green = np.zeros(Kd_490.shape)
      solar_red = np.zeros(Kd_490.shape)

      lunar_490nm = np.zeros(Kd_490.shape)
      lunar_blue = np.zeros(Kd_490.shape)
      lunar_green = np.zeros(Kd_490.shape)
      lunar_red = np.zeros(Kd_490.shape)

      for ilat in range(len(lats)):
         location = EarthLocation(lat=lats[ilat], lon=longitude_deg, height=1)            
         # For solar use the 15th day of the calendar month at midday to calculate surface irradiance
         solar_date = datetime.datetime(year, month, solar_day, solar_hour, minute, tzinfo=datetime.timezone.utc) # Create date - accurate to minute incr.
         jday = solar_date.timetuple().tm_yday # Retrieve date as a tuple
         H = ((solar_hour+minute)/24)
         dday = int(jday)+H # Combine times as a decimal day
         Ecc = (1.0+1.67E-2*np.cos(2.0*np.pi*float(dday-3)/365.0))**2 # correction for eccentricity of Earth's orbit
         altitude_deg = get_altitude(lats[ilat], longitude_deg, solar_date) # Retrieve altitude of the sun

         if (altitude_deg > 0.):
            #airmass = get_air_mass_kasten_young(altitude_deg)
            # Run a more sophisticated atmospheric model
            df_dir_dif = gcirrad.dir_dif_irradiance(df_trans_atmos, altitude_deg, Ecc)
            # Surface values of the SolSpec
            bin1 = df_dir_dif[df_dir_dif['Wavelength']==485].index[0]
            bin2 = df_dir_dif[df_dir_dif['Wavelength']==495].index[0]
            I = df_dir_dif['Ed'].to_numpy() # convert intensity column to array
            x = df_dir_dif['Wavelength'].to_numpy() # convert index column to array
            bi = I[bin1:bin2] # store interval of intensity
            bx = x[bin1:bin2] # store interval of index (wavelength)
            # Integrate for 10nm bin
            Int = (scipy.integrate.simps(bi,bx, dx=0.5))
            solar_490nm[ilat,0:len(lons)-1] = Int

            SolSpecSurface = SpectralSplit.spectral_split('Wavelength','Ed',df_dir_dif)
            #PARSurface = SolSpecSurface[['PAR']].to_numpy()[0]
            SolSpecSurface = SolSpecSurface[['Red','Green','Blue']]
            solar_blue[ilat,0:len(lons)-1] = SolSpecSurface['Blue'][0]
            solar_green[ilat,0:len(lons)-1] = SolSpecSurface['Green'][0]
            solar_red[ilat,0:len(lons)-1] = SolSpecSurface['Red'][0]
         
         # for Lunar use the date of the full moon and where moon reaches culmination at Greenwich Meridian
         lunar_date = datetime.datetime(year, month, full_moon_dates[month - 1], full_moon_hours[month - 1], full_moon_minutes[month - 1], tzinfo=datetime.timezone.utc)
         dt_lunar = Time(lunar_date)
         EarthMoon = get_moon(dt_lunar, location) # Relative positioning              
         moon_icrs = EarthMoon.transform_to('icrs') # Relative Ephemeris calculation    
         moonaltaz = moon_icrs.transform_to(AltAz(obstime=dt_lunar, location=location)) # Transform positioning to Alt, Az
         alt_lunar = float(Angle(moonaltaz.alt).degree) # Lunar Altitude 
         az_lunar = float(Angle(moonaltaz.az).degree) # Lunar Azimuth                     
         Phase = float(moon.moon_illumination(dt_lunar)) # Lunar Phase 
         Phase_lb = lumme_bowell(moon.moon_phase_angle(dt_lunar).value) # Lunar Phase function - Lumme-Bowell
         albedo_phased = df_albedo_lunar['Broadband'].to_numpy()[0]*Phase_lb

         if (alt_lunar > 0.):
            # Use the Gregg and Carder atmospheric model, with correction for the lunar spectral albedo (lunar=True)
            # and the lunar phase (multiplied by the lunar solid view angle)
            df_dir_dif = gcirrad.dir_dif_irradiance(df_trans_atmos, alt_lunar, Ecc, lunar=True, phase_sva=Phase_lb*lun_sva)
            # Surface values of the LunSpec
            bin1 = df_dir_dif[df_dir_dif['Wavelength']==485].index[0]
            bin2 = df_dir_dif[df_dir_dif['Wavelength']==495].index[0]
            I = df_dir_dif['Ed'].to_numpy() # convert intensity column to array
            x = df_dir_dif['Wavelength'].to_numpy() # convert index column to array
            bi = I[bin1:bin2] # store interval of intensity
            bx = x[bin1:bin2] # store interval of index (wavelength)
            # Integrate for 10nm bin
            Int = (scipy.integrate.simps(bi,bx, dx=0.5))
            lunar_490nm[ilat,0:len(lons)-1] = Int*1000000. # convert from W/m^2 to uW/m^2

            LunSpecSurface = SpectralSplit.spectral_split('Wavelength','Ed',df_dir_dif)
            LunSpecSurface = LunSpecSurface[['Red','Green','Blue']]*1000000. # convert from W/m^2 to uW/m^2

            lunar_blue[ilat,0:len(lons)-1] = LunSpecSurface['Blue'][0]
            lunar_green[ilat,0:len(lons)-1] = LunSpecSurface['Green'][0]
            lunar_red[ilat,0:len(lons)-1] = LunSpecSurface['Red'][0]
            
         # suggest that need to put check in here for 50N to see what lunar geometry is
         # then check against web position
         if (lats[ilat] > 50.0 and lats[ilat] < 50.1):
            print("Solar:", lats[ilat], year, month, solar_day, solar_hour, altitude_deg)
            print("Lunar:", lats[ilat], year, month, full_moon_dates[month - 1], full_moon_hours[month - 1], full_moon_minutes[month - 1], alt_lunar, az_lunar, Phase)

      # write the output to a netCDF file
      print(" Writing output for month: ", month_str[month-1])
      print("  Filename: ", ofname)
      
      # output the resultant netCDF file containing Kd(RGB) and lat, lon
      w_nc_fid = nc.Dataset(outdir+ofname, 'w', format='NETCDF4')
      w_nc_fid.description = "Spectral Solar and Lunar surface irradiance values modelled for each month (2022)"
      
      # Create the new dimensions from the resized variables
      data = {}
      new_dims = [Kd_490_shape[0], Kd_490_shape[1]]
      
      new_nc_dims = ['lat','lon']
      counter = 0
      for dim in new_nc_dims:
         #w_nc_fid.createDimension(dim, infile.variables[dim].size)
         w_nc_fid.createDimension(dim, new_dims[counter])
         data[dim] = w_nc_fid.createVariable(dim, infile.variables[dim].dtype,(dim,))
         # You can do this step yourself but someone else did the work for us.
         #pdb.set_trace()
         
         print(infile.variables[dim].ncattrs())
         for ncattr in infile.variables[dim].ncattrs():
            if (ncattr != '_FillValue'):
               data[dim].setncattr(ncattr, infile.variables[dim].getncattr(ncattr))
            else:
               if (dim == 'lon'):
                  data[dim].setncattr('axis', "X")
               if (dim == 'lat'):
                  data[dim].setncattr('axis', "Y")

         counter = counter+1

      w_nc_fid.variables['lat'][:] = lats
      w_nc_fid.variables['lon'][:] = lons
      
      # Create output variables
      # Lunar
      w_nc_var = w_nc_fid.createVariable('Lunar_Irradiance_490', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Lunar Irradiance at 490 nm",\
                    'units': u"uW/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Lunar Irradiance at full moon at culmination (490 nm): uW/m2"})
      w_nc_fid.variables['Lunar_Irradiance_490'][:] = lunar_490nm

      # Blue Lunar
      w_nc_var = w_nc_fid.createVariable('Lunar_Irradiance_Blue', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Lunar Irradiance - Blue",\
                    'units': u"uW/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Lunar Irradiance at full moon at culmination (Blue: 400 - 500 nm): uW/m2"})
      w_nc_fid.variables['Lunar_Irradiance_Blue'][:] = lunar_blue

      # Green Lunar
      w_nc_var = w_nc_fid.createVariable('Lunar_Irradiance_Green', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Lunar Irradiance - Green",\
                    'units': u"uW/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Lunar Irradiance at full moon at culmination (Green: 495 - 560 nm): uW/m2"})
      w_nc_fid.variables['Lunar_Irradiance_Green'][:] = lunar_green

      # Red Lunar
      w_nc_var = w_nc_fid.createVariable('Lunar_Irradiance_Red', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Lunar Irradiance - Red",\
                    'units': u"uW/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Lunar Irradiance at full moon at culmination (Red: 620 - 740 nm): uW/m2"})
      w_nc_fid.variables['Lunar_Irradiance_Red'][:] = lunar_red

      # Solar
      w_nc_var = w_nc_fid.createVariable('Solar_Irradiance_490', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Solar Irradiance at 490 nm",\
                    'units': u"W/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Surface Solar Irradiance at Zenith(490 nm): W/m2"})
      w_nc_fid.variables['Solar_Irradiance_490'][:] = solar_490nm

      # Blue Solar
      w_nc_var = w_nc_fid.createVariable('Solar_Irradiance_Blue', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Solar Irradiance - Blue",\
                    'units': u"W/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Solar Irradiance at zenith (Blue: 400 - 500 nm): W/m2"})
      w_nc_fid.variables['Solar_Irradiance_Blue'][:] = solar_blue

      # Green Solar
      w_nc_var = w_nc_fid.createVariable('Solar_Irradiance_Green', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Solar Irradiance - Green",\
                    'units': u"W/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Solar Irradiance at zenith (Green: 495 - 560 nm): W/m2"})
      w_nc_fid.variables['Solar_Irradiance_Green'][:] = solar_green

      # Red 
      w_nc_var = w_nc_fid.createVariable('Solar_Irradiance_Red', 'f8', ('lat','lon'))
      w_nc_var.setncatts({'long_name': u"Solar Irradiance - Red",\
                    'units': u"W/m2", 'level_desc': u'Surface',\
                    'var_desc': u"Solar Irradiance at zenith (Red: 620 - 740 nm): W/m2"})
      w_nc_fid.variables['Solar_Irradiance_Red'][:] = solar_red
      
      w_nc_fid.close()  # close the output file

   infile.close()
      
   sys.exit()
      
if __name__=='__main__':
   main()
