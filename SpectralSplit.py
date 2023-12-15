# Spectral range of Blue Green and Red light are taken from Davies et al. (2020) [hydrolight model]
# Read data file: Solar NERL downloaded spectral measuremnets
# Read data file: Lunar provided by Dr. Tim Smyth 
# Read data file: ALAN provided by Dr. Tim Smyth
# Unit conversion http://www.egc.com/useful_info_lighting.php?input=&source=Low-pressure%20sodium&text=&units=Photons+To+W%2Fm%3Csup%3E%3Cfont+size%3D%22-2%22%3E2%3C%2Ffont%3E%3C%2Fsup%3E
### Data file must be in the same directory as the script. ###
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate
import numpy as np
import pdb

######################################################################################################
######################################################################################################

def sol_spectra(fname_Sol):
    df = pd.read_csv(fname_Sol, skiprows=1) # Read csv with filename specified in model script
    PAR = np.array(np.arange(400, 700, 10)) # set array length and binwidth to every 10nm
    d = pd.DataFrame(columns=PAR) # create dataframe with columns of wavelength name
    PAR_int = np.array([])
    
    for i in range(len(PAR)):
        
        lim1 = PAR[i]-1 # define lower limit
        lim2 = PAR[i]+10 # define upper limit
        bin1 = df[df['Wvlgth nm']==lim1].index[0] # search for lower limit value in .csv
        bin2 = df[df['Wvlgth nm']==lim2].index[0] # search for upper limit value in .csv
        I = df['Etr W*m-2*nm-1'].to_numpy() # convert intensity column to array
        x = df['Wvlgth nm'].to_numpy() # convert index column to array
        bi = I[bin1:bin2] # store interval of intensity
        bx = x[bin1:bin2] # store interval of index (wavelength)
        # Integrate for each bin of 10nm, Multiply by 2.022 to convert W/m^2 (unit of vals in .csv) to umol/m^2/s
        Int = (scipy.integrate.simps(bi,bx, dx=0.5))
        
        PAR_int = np.append(PAR_int, Int) # Append to respective column in array
    print('Solar Spectral sum', sum(PAR_int))
    d = d.append(pd.Series(PAR_int, index=d.columns), ignore_index=True) # convert array to dataframe with specified column names 

    ############################# Plot Solar Spectra ##########################################              
    # fig, ax = plt.subplots(1)
    # ax.plot(df.iloc[:,0], df.iloc[:,1], color='goldenrod'); ax.set_xlim([150,760])
    # ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Irradiance (W/m$^{-2}$)'); ax.set_title('Solar Spectral Split')
    return d

def lun_spectra(fname_Lun):
    df = pd.read_csv(fname_Lun) # Read csv with filename specified in model script
    PAR = np.array(np.arange(400, 700, 10)) # set array length and binwidth to every 10nm
    e = pd.DataFrame(columns=PAR) # create dataframe with columns of wavelength name
    PAR_int = np.array([])
    
    for i in range(len(PAR)):
        
        lim1 = PAR[i]-1 # define lower limit
        # print(lim1, 'lim1')
        lim2 = PAR[i]+10 # define upper limit
        # print(lim2)
                
        bin1 = df[df['Wavelength']==lim1].index[0] # search for lower limit value in .csv
        bin2 = df[df['Wavelength']==lim2].index[0] # search for upper limit value in .csv
        I = df['Moon(W/m2/nm)'].to_numpy() # convert intensity column to array
        x = df['Wavelength'].to_numpy() # convert index column to array
        bi = I[bin1:bin2] # store interval of intensity
        bx = x[bin1:bin2] # store interval of index (wavelength)
        # Integrate for each bin of 10nm, Multiply by 2.022 to convert W/m^2 (unit of vals in .csv) to umol/m^2/s
        Int = (scipy.integrate.simps(bi,bx, dx=0.5))
        PAR_int = np.append(PAR_int, Int) # Append to respective column in array
    print('Lunar Spectral sum', sum(PAR_int))
    e = e.append(pd.Series(PAR_int, index=e.columns), ignore_index=True) # convert array to dataframe with specified column names        

    ############################# Plot lunar Spectra ##########################################          
    # fig, ax = plt.subplots(1)
    # ax.plot(df.iloc[:,0], df.iloc[:,1], color='grey'); ax.set_xlim([150,760])
    # ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Irradiance (W/m$^{-2}$)'); ax.set_title('Lunar Spectral Split')
    return e
    
def A_spectra(fname_ALAN):
    df = pd.read_csv(fname_ALAN, skiprows=1) # skiprows: insert number of rows before header row, headers row is skipped automatically
    PAR = np.array(np.arange(400, 700, 10)) # set array length and binwidth to every 10nm
    f = pd.DataFrame(columns=PAR) # create dataframe with columns of wavelength name
    PAR_int = np.array([])

    for i in range(len(df)):
        df.iloc[i,1] = float(df.iloc[i,1])


    for i in range(len(PAR)):
        lim1 = PAR[i]-1 # define lower limit
        lim2 = PAR[i]+10 # define upper limit
        
        bin1 = df[df.iloc[:,0]==lim1].index[0] # search for lower limit value in .csv
        bin2 = df[df.iloc[:,0]==lim2].index[0] # search for upper limit value in .csv
        I = df.iloc[:,1].to_numpy() # convert intensity column to array
        x = df.iloc[:,0].to_numpy() # convert index column to array
        bi = I[bin1:bin2]# store interval of intensity
        bx = x[bin1:bin2]# store interval of index (wavelength)
        # Integrate for each bin of 10nm, Multiply by 2.022 to convert W/m^2 (unit of vals in .csv) to umol/m^2/s 
        Int = (scipy.integrate.simps(bi,bx, dx=0.5))
        if Int < 0: 
            Int = 0
        PAR_int = np.append(PAR_int, Int) # Append to respective column in array
    print('ALAN Spectral sum', sum(PAR_int))
    f = f.append(pd.Series(PAR_int, index=f.columns), ignore_index=True) # convert array to dataframe with specified column names
    
    ############################# Plot ALAN Spectra ##########################################  
    # fig, ax = plt.subplots(1)
    # ax.plot(df.iloc[:,0], df.iloc[:,1], color='dodgerblue'); ax.set_xlim([150,760])
    # ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('Irradiance (W/m$^{-2}$)'); ax.set_title('ALAN Spectral Split')
    return f

########################################################################################################
########################################################################################################
def spectra_run(fname_Sol, fname_Lun, fname_ALAN, input_flag):
    df = pd.read_csv(fname_Sol, skiprows=1)
    DF = pd.read_csv(fname_Lun)
    Df = pd.read_csv(fname_ALAN) # skiprows: insert number of rows before header row, headers row is skipped automatically

    print("Running Solar Spectra")
    Sol = spectral_split('Wvlgth nm', 'Etr W*m-2*nm-1', df)
    print("Running Lunar Spectra")
    Lun = spectral_split('Wavelength', 'Moon(W/m2/nm)', DF)
    print("Running LED Spectra")
    if input_flag == 1:
        print("Select a type of lamp: \nA: HPS(W/m-2)\nB: LPS(W/m-2)\nC: MH(W/m-2)\nD: LED(W/m-2)")
        INPUT_ALAN_TYPE = input()
        if INPUT_ALAN_TYPE=="A":
            ALAN = spectral_split('Wavelength', 'HPS(W/m-2)', Df)
            ALAN_TYPE = "HPS"
        elif INPUT_ALAN_TYPE=="B":
            ALAN = spectral_split('Wavelength', 'LPS(W/m-2)', Df)
            ALAN_TYPE = "LPS"
        elif INPUT_ALAN_TYPE=="C":
            ALAN = spectral_split('Wavelength', 'MH(W/m-2)', Df)
            ALAN_TYPE = "MH"
        elif INPUT_ALAN_TYPE=="D":
            ALAN = spectral_split('Wavelength', 'LED(W/m-2)', Df)
            ALAN_TYPE = "LED"
    else:
        ALAN = spectral_split('Wavelength', 'LED(W/m-2)', Df)
        ALAN_TYPE = "LED"
   
    return Sol, Lun, ALAN, ALAN_TYPE
    

def spec_frame(PAR, R, G, B): 
    columns = ['PAR', 'Red', 'Green', 'Blue']
    data = {'PAR': [PAR], 'Red': [R], 'Green': [G], 'Blue': [B]}
    df = pd.DataFrame(data, columns=columns)
    return df

#   Model for atmospheric transmittance of solar irradiance through
#   a maritime atmosphere.  Computes direct and diffuse separately.
#   Includes water vapor and oxygen absorption.
def dir_dif_irradiance(wavelength_col, F0_col, a_oz_col, a_ox_col, a_w_col, df, theta):

#      subroutine atmodd(iblw,rad,lam,theta,oza,ag,aw,sco3,p,wv,
#    * rh,am,wsm,ws,vis,Fo,Edir,Edif,Ed)
#
#      parameter(nlt=351)
#      integer lam(nlt)
#      real Fo(nlt),oza(nlt),ag(nlt),Edir(nlt),Edif(nlt),Ed(nlt)
#      real aw(nlt)
#      double precision rad

    p0 = 1013.25 # standard pressure in mb
    p = 1013.25  # assume standard pressure in mb

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
#      call navaer(rh,am,wsm,ws,vis,beta,alpha,wa,asymp)
    rlam = 0.55 # reference wavelength (in microns)
    rh = 80.0 # Typical humidities
    rnum = 2.0 - rh/100.0
    rden = 6.0*(1.0-rh/100.0)
    frh = (rnum/rden)**0.333
     

#      write(6,*)
#      write(6,*)'   Aerosol parameters from the Navy aerosol model:'
#      write(6,*)'     alpha                    = ',alpha
#      write(6,*)'     beta                     = ',beta
#      write(6,*)'     asymmetry parameter      = ',asymp
#      write(6,*)'     single scattering albedo = ',wa
#      eta = -alpha
#   Forward scattering probability
#      alg = alog(1.0-asymp)
#      afs = alg*(1.459+alg*(.1595+alg*.4129))
#      bfs = alg*(.0783+alg*(-.3824-alg*.5874))
#      Fa = 1.0 - 0.5*exp((afs+bfs*cosunz)*cosunz)

    pdb.set_trace()

#  Surface reflectance (assume dark surface)
    rod = 0.0
    ros = 0.0

#  Compute spectral irradiance
#      do nl = 1,nlt
#    Rayleigh, by Bird's method
#       rlam = lam(nl)*1.0E-3
#       tr = 1.0/(115.6406*rlam**4 - 1.335*rlam**2)
#       rtra = exp(-tr*rmp)   !transmittance
#    Ozone
#       to = oza(nl)*sco3   !optical thickness
#       otra = exp(-to*rmo)   !transmittance
#   Aerosols
#       ta = beta*rlam**eta
#       if (lam(nl) .eq. 550)then
#	write(6,*)'   Aerosol optical thickness at 550 nm = ',ta
#       endif
#       atra = exp(-ta*rm)
#       taa = exp(-(1.0-wa)*ta*rm)
#       tas = exp(-wa*ta*rm)
#   Oxygen/gases
#       gtmp = (1.0 + 118.3*ag(nl)*rmp)**0.45
#       gtmp2 = -1.41*ag(nl)*rmp
#       gtra = exp(gtmp2/gtmp)
#   Water Vapor
#       wtmp = (1.0+20.07*aw(nl)*wv*rm)**0.45
#       wtmp2 = -0.2385*aw(nl)*wv*rm
#       wtra = exp(wtmp2/wtmp)
#
#  Direct irradiance
#       Edir(nl) = Fo(nl)*cosunz*rtra*otra*atra*gtra*wtra*(1.0-rod)
#
#   Diffuse irradiance
#       dray = Fo(nl)*cosunz*gtra*wtra*otra*taa*0.5*
#     *      (1.0-rtra**.95)
#       daer = Fo(nl)*cosunz*gtra*wtra*otra*rtra**1.5*taa*Fa*
#     *      (1.0-tas)
#
#  Total diffuse
#       Edif(nl) = (dray + daer)*(1.0-ros)

#       Ed(nl) = Edir(nl) + Edif(nl)


    return df


# Determine the atmospheric diffuse attenuation coefficient
def k_spectral_split(wavelength_col, trans_col, df):
    # Thresholds used in the Davies et al, 2020 paper " Biologically active ALAN at seafloor"
    bbin1 = df[df[wavelength_col]==400].index[0] # Broadband lower limit index
    bbin2 = df[df[wavelength_col]==700].index[0] # Broadband upper limit index
    
    bin1 = df[df[wavelength_col]==400].index[0] # Blue lower limit index
    bin2 = df[df[wavelength_col]==500].index[0] # Blue upper limit index

    gin1 = df[df[wavelength_col]==495].index[0] # Green lower limit index
    gin2 = df[df[wavelength_col]==560].index[0] # Green upper limit index

    rin1 = df[df[wavelength_col]==620].index[0] # Red lower limit index
    rin2 = df[df[wavelength_col]==740].index[0] # Red upper limit index 

    t = df[trans_col].to_numpy()
    x = df[wavelength_col].to_numpy()
    
    BBt = t[bbin1:bbin2]
    BBx = x[bbin1:bbin2]
    Bt = t[bin1:bin2]
    Bx = x[bin1:bin2]
    Gt = t[gin1:gin2]
    Gx = x[gin1:gin2]
    Rt = t[rin1:rin2]
    Rx = x[rin1:rin2]
    
    k_atmos_bb = -np.log10(np.mean(BBt))
    k_atmos_b = -np.log10(np.mean(Bt))
    k_atmos_g = -np.log10(np.mean(Gt))
    k_atmos_r = -np.log10(np.mean(Rt))

    columns = ['Broadband', 'Red', 'Green', 'Blue']
    data = {'Broadband': [k_atmos_bb], 'Red': [k_atmos_r], 'Green': [k_atmos_g], 'Blue': [k_atmos_b]}
    dataframe = pd.DataFrame(data, columns=columns)

    return dataframe

# Determine the spectral lunar albedo in predefined bands
def lunar_albedo_spectral_split(wavelength_col, albedo_col, df):
    # Thresholds used in the Davies et al, 2020 paper " Biologically active ALAN at seafloor"
    BB_df = df[df[wavelength_col].between(400,700)]
    BBa = BB_df['AverageMoonAlbedo']

    B_df = df[df[wavelength_col].between(400,500)]
    Ba = B_df['AverageMoonAlbedo']

    G_df = df[df[wavelength_col].between(495,560)]
    Ga = G_df['AverageMoonAlbedo']
    
    R_df = df[df[wavelength_col].between(620,740)]
    Ra = R_df['AverageMoonAlbedo']

    lunar_albedo_bb = np.mean(BBa)
    lunar_albedo_b = np.mean(Ba)
    lunar_albedo_g = np.mean(Ga)
    lunar_albedo_r = np.mean(Ra)

    columns = ['Broadband', 'Red', 'Green', 'Blue']
    data = {'Broadband': [lunar_albedo_bb], 'Red': [lunar_albedo_r], 'Green': [lunar_albedo_g], 'Blue': [lunar_albedo_b]}
    dataframe = pd.DataFrame(data, columns=columns)

    return dataframe



def spectral_split(wavelength_col, intensity_col, df):
    
    # ALAN dataset records -ve values of intensity when wavelength < 400nm
    ## SOLAR Spectra

    # Thresholds used in the Davies et al, 2020 paper " Biologically active ALAN at seafloor"
    bbin1 = df[df[wavelength_col]==400].index[0] # Broadband lower limit index
    bbin2 = df[df[wavelength_col]==700].index[0] # Broadband upper limit index

    bin1 = df[df[wavelength_col]==400].index[0] # Blue lower limit index
    bin2 = df[df[wavelength_col]==500].index[0] # Blue upper limit index

    gin1 = df[df[wavelength_col]==495].index[0] # Green lower limit index
    gin2 = df[df[wavelength_col]==560].index[0] # Green upper limit index

    rin1 = df[df[wavelength_col]==620].index[0] # Red lower limit index
    rin2 = df[df[wavelength_col]==740].index[0] # Red upper limit index 

    i = df[intensity_col].to_numpy()
    x = df[wavelength_col].to_numpy()
    
    BBi = i[bbin1:bbin2]
    BBx = x[bbin1:bbin2]
    Bi = i[bin1:bin2]
    Bx = x[bin1:bin2]
    Gi = i[gin1:gin2]
    Gx = x[gin1:gin2]
    Ri = i[rin1:rin2]
    Rx = x[rin1:rin2]

    PARLight = (scipy.integrate.simps(BBi,BBx, dx=0.5))
    BlueLight = (scipy.integrate.simps(Bi,Bx, dx=0.5))
    GreenLight = (scipy.integrate.simps(Gi,Gx, dx=0.5))
    RedLight = (scipy.integrate.simps(Ri,Rx, dx=0.5))
    dataframe = spec_frame(PARLight, RedLight, GreenLight, BlueLight)
    # pdb.set_trace()
    #print('Irr. (W/m^2), '
    #      'Blue:', BlueLight,
    #      'Green:', GreenLight,
    #      'Red: ', RedLight,
    #      'PAR(400 - 700nm): ', PARLight)
    return dataframe

##########################################################################################
##########################################################################################

def Individual_splits(fname_Sol, fname_Lun, fname_ALAN):
    df = pd.read_csv(fname_Sol, skiprows=1)
    DF = pd.read_csv(fname_Lun)
    Df = pd.read_csv(fname_ALAN) # skiprows: insert number of rows before header row, headers row is skipped automatically

    bin1 = df[df['Wvlgth nm']==400].index[0] # Blue lower limit index
    bin2 = df[df['Wvlgth nm']==480].index[0] # Blue upper limit index

    gin1 = df[df['Wvlgth nm']==490].index[0] # Green lower limit index
    gin2 = df[df['Wvlgth nm']==590].index[0] # Green upper limit index

    rin1 = df[df['Wvlgth nm']==600].index[0] # Red lower limit index
    rin2 = df[df['Wvlgth nm']==700].index[0] # Red upper limit index

    i = df['Etr W*m-2*nm-1'].to_numpy()
    x = df['Wvlgth nm'].to_numpy()
    Bi = i[bin1:bin2]
    Bx = x[bin1:bin2]
    Gi = i[gin1:gin2]
    Gx = x[gin1:gin2]
    Ri = i[rin1:rin2]
    Rx = x[rin1:rin2]

    Sol_BlueLight = (scipy.integrate.simps(Bi,Bx, dx=0.5))
    Sol_GreenLight = (scipy.integrate.simps(Gi,Gx, dx=0.5))
    Sol_RedLight = (scipy.integrate.simps(Ri,Rx, dx=0.5))
    print('Solar Irr. (W/m^2), '
        'Blue:', Sol_BlueLight,
        'Green:', Sol_GreenLight,
        'Red: ', Sol_RedLight)



        ## LUNAR Spectra

    BIN1 = DF[DF['Wavelength']==400].index[0] # Blue lower limit index
    BIN2 = DF[DF['Wavelength']==480].index[0] # Blue upper limit index

    GIN1 = DF[DF['Wavelength']==490].index[0] # Green lower limit index
    GIN2 = DF[DF['Wavelength']==590].index[0] # Green upper limit index

    RIN1 = DF[DF['Wavelength']==600].index[0] # Red lower limit index
    RIN2 = DF[DF['Wavelength']==700].index[0] # Red upper limit index


    I = DF['Moon(W/m2/nm)'].to_numpy()
    print(I)
    X = DF['Wavelength'].to_numpy()
    BI = I[BIN1:BIN2]
    print(BI)
    BX = X[BIN1:BIN2]
    GI = I[GIN1:GIN2]
    GX = X[GIN1:GIN2]
    RI = I[RIN1:RIN2]
    RX = X[RIN1:RIN2]

    Lun_BlueLight = (scipy.integrate.simps(BI,BX, dx=0.5))
    Lun_GreenLight = (scipy.integrate.simps(GI,GX, dx=0.5))
    Lun_RedLight = (scipy.integrate.simps(RI,RX, dx=0.5))

    print('Lunar Irr. (W/m^2), '
        'Blue:', Lun_BlueLight,
        'Green:', Lun_GreenLight,
        'Red: ', Lun_RedLight)

    Bin1 = Df[Df['Wavelength']==400].index[0] # Blue lower limit index
    Bin2 = Df[Df['Wavelength']==480].index[0] # Blue upper limit index

    Gin1 = Df[Df['Wavelength']==490].index[0] # Green lower limit index
    Gin2 = Df[Df['Wavelength']==590].index[0] # Green upper limit index

    Rin1 = Df[Df['Wavelength']==600].index[0] # Red lower limit index
    Rin2 = Df[Df['Wavelength']==700].index[0] # Red upper limit index

    i = Df['LED  (W/m-2)'].to_numpy()
    print(i)
    x = Df['Wavelength'].to_numpy()
    Bi = i[Bin1:Bin2]
    print(Bi)
    Bx = x[Bin1:Bin2]
    Gi = i[Gin1:Gin2]
    Gx = x[Gin1:Gin2]
    Ri = i[Rin1:Rin2]
    Rx = x[Rin1:Rin2]

    ALAN_BlueLight = (scipy.integrate.simps(Bi,Bx, dx=0.5))
    ALAN_GreenLight = (scipy.integrate.simps(Gi,Gx, dx=0.5))
    ALAN_RedLight = (scipy.integrate.simps(Ri,Rx, dx=0.5))
    print('ALAN Irr. (W/m^2), '
        'Blue:', ALAN_BlueLight,
        'Green:', ALAN_GreenLight,
        'Red: ', ALAN_RedLight)

    return Sol_RedLight, Sol_GreenLight, Sol_BlueLight, Lun_RedLight, Lun_GreenLight, Lun_BlueLight


if __name__ == '__main__':
    fname_Sol = 'SolarSpectra.csv'
    fname_Lun = 'moon_spectra.csv'
    fname_ALAN = 'LEDSpectra.csv'
    Individual_splits(fname_Sol, fname_Lun, fname_ALAN)


