#!/usr/bin/env python3
import numpy as np
import pandas as pd
import pdb
import os

# Just return the TOA irradiances
def TOA_irradiance(df_gcirrad, Ecc):
    lam = df_gcirrad['Wavelength'].to_numpy()
    Hobar = df_gcirrad['F0'].to_numpy()
    Fobar = Hobar*10.0   #convert to W/m2/nm
    Fo = Fobar*Ecc
    columns = ['Wavelength', 'Fo']
    data = {'Wavelength': lam, 'Fo': Fo}
    dataframe = pd.DataFrame(data, columns=columns)
    
    return dataframe
    

#   Model for atmospheric transmittance of solar irradiance through
#   a maritime atmosphere.  Computes direct and diffuse separately.
#   Includes water vapor and oxygen absorption.
#   Addition of lunar component
def dir_dif_irradiance(df_gcirrad, altitude_deg, Ecc, lunar=False, phase_sva=1.0):
    theta = 90. - altitude_deg # zenith angle in degrees

    p0 = 1013.25 # standard pressure in mb
    p = 1013.25  # assume standard pressure in mb
    to3 = 300.   # Ozone concentration (Dobson Units)
    sco3 = to3*1.0e-3
    wv = 1.5     # precipitable water (water vapor) in cm
    
#   Compute atmospheric path lengths (air mass); pressure-corrected
    cosunz = np.cos(np.radians(theta))
#   Kasten and Young 1989.
    rex = -1.253
    rtmp = (93.885-theta)**rex
    rm = 1.0/(cosunz+0.15*rtmp)
    rex = -1.6364
    rtmp = (96.07995-theta)**rex
    rm = 1.0/(cosunz+0.50572*rtmp)
    rmp = p/p0*rm
    otmp = (cosunz*cosunz+44.0/6370.0)**0.5
    rmo = (1.0+22.0/6370.0)/otmp
    
#  Aerosol parameters; simplified Navy aerosol model
    wa, alpha, beta, asymp = navaer()

    eta = -alpha
#   Forward scattering probability
    alg = np.log(1.0-asymp)
    afs = alg*(1.459+alg*(.1595+alg*.4129))
    bfs = alg*(.0783+alg*(-.3824-alg*.5874))
    Fa = 1.0 - 0.5*np.exp((afs+bfs*cosunz)*cosunz)

#  Surface reflectance (assume dark surface)
    rod = 0.0
    ros = 0.0

#  Compute spectral irradiance
    lam = df_gcirrad['Wavelength'].to_numpy()
    Hobar = df_gcirrad['F0'].to_numpy()
    oza = df_gcirrad['a_oz'].to_numpy()
    ag = df_gcirrad['a_ox'].to_numpy()
    aw = df_gcirrad['a_w'].to_numpy()
    lunar_albedo = df_gcirrad['LunarAlbedo'].to_numpy()
    
    Fobar = Hobar*10.0   #convert to W/m2/nm
    #Fo = Fobar*Ecc*lunfactor
    if (lunar):
       Fo = Fobar*Ecc*lunar_albedo*phase_sva
    else:
       Fo = Fobar*Ecc
    
    Ed = np.zeros(len(lam))
    Edir = np.zeros(len(lam))
    Edif = np.zeros(len(lam))

    for nl in range(len(lam)):
#      Rayleigh, by Bird's method
       rlam = lam[nl]*1.0e-3
       tr = 1.0/(115.6406*rlam**4 - 1.335*rlam**2)
       rtra = np.exp(-tr*rmp)   #transmittance
#      Ozone
       to = oza[nl]*sco3   #optical thickness
       otra = np.exp(-to*rmo)   #transmittance
#      Aerosols
       ta = beta*rlam**eta
#       if lam[nl] == 550:
#          print('Aerosol optical thickness at 550 nm = ',ta)
       atra = np.exp(-ta*rm)
       taa = np.exp(-(1.0-wa)*ta*rm)
       tas = np.exp(-wa*ta*rm)
#      Oxygen/gases
       gtmp = (1.0 + 118.3*ag[nl]*rmp)**0.45
       gtmp2 = -1.41*ag[nl]*rmp
       gtra = np.exp(gtmp2/gtmp)
#      Water Vapor
       wtmp = (1.0+20.07*aw[nl]*wv*rm)**0.45
       wtmp2 = -0.2385*aw[nl]*wv*rm
       wtra = np.exp(wtmp2/wtmp)
#
#      Direct irradiance
       Edir[nl] = Ecc*Fo[nl]*cosunz*rtra*otra*atra*gtra*wtra*(1.0-rod)
#
#      Diffuse irradiance
       dray = Ecc*Fo[nl]*cosunz*gtra*wtra*otra*taa*0.5*(1.0-rtra**.95)
       daer = Ecc*Fo[nl]*cosunz*gtra*wtra*otra*rtra**1.5*taa*Fa*(1.0-tas)
#
#      Total diffuse
       Edif[nl] = (dray + daer)*(1.0-ros)
#
       Ed[nl] = Edir[nl] + Edif[nl]

    columns = ['Wavelength', 'Fo', 'Edir', 'Edif', 'Ed']
    data = {'Wavelength': lam, 'Fo': Fo,'Edir': Edir, 'Edif': Edif, 'Ed': Ed}
    dataframe = pd.DataFrame(data, columns=columns)

    return dataframe
    
def navaer():
#  Computes aerosol parameters according to a simplified version
#  of the Navy marine aerosol model.
#
   ro = [0.03,0.24,2.0]
   r = [0.1,1.0,10.0]
   a = np.zeros(3)
   dndr = np.zeros(3)
   
   
   rlam = 0.55
   rh = 80.0 # relative humidity
   am = 5.0  # Marine aerosol model
   wsm = 5.0 # mean wind speed
   ws = 5.0  # wind speed
   vis = 15.0 # average visibility (km)

   if rh >= 100.0:
      rh = 99.9
   rnum = 2.0 - rh/100.0
   rden = 6.0*(1.0-rh/100.0)
   frh = (rnum/rden)**0.333

#  Size distribution amplitude components
   a[0] = 2000.0*am*am
   a[1] = 5.866*(wsm-2.2)
   if a[1] < 0.5:
      a[1] = 0.5
   a[2] = 0.01527*(ws-2.2)*0.05   #from Hughes 1987
   if a[2] < 1.4e-5:
      a[2] = 1.4e-5

#  Compute size distribution at three selected radii according to
#  Navy method
   #pdb.set_trace()
   for n in range(len(ro)):
      dndr[n] = 0.0
      for i in range(len(ro)):
         rden = frh*ro[i]
         arg = np.log(r[n]/rden)*np.log(r[n]/rden)
         rval = a[i]*np.exp(-arg)/frh
         dndr[n] = dndr[n] + rval
#
#  Least squares approximation
   sumx = 0.0
   sumy = 0.0
   sumxy = 0.0
   sumx2 = 0.0
   for n in range(len(ro)):
      sumx = sumx + np.log10(r[n])
      sumy = sumy + np.log10(dndr[n])
      sumxy = sumxy + np.log10(r[n])*np.log10(dndr[n])
      sumx2 = sumx2 + np.log10(r[n])*np.log10(r[n])
   gama = sumxy/sumx2
   rlogc = sumy/3.0 - gama*sumx/3.0
   alpha = -(gama+3.0)
#
#  Compute beta
   cext = 3.91/vis
   beta = cext*rlam**alpha

#  Compute asymmetry parameter -- a function of alpha
   if alpha > 1.2:
      asymp = 0.65
   elif alpha < 0.0:
      asymp = 0.82
   else:
      asymp = -0.14167*alpha + 0.82

#  Single scattering albedo at 550; function of relative humidity
   wa = (-0.0032*am + 0.972)*np.exp(3.06e-4*rh)

   return wa, alpha, beta, asymp 

