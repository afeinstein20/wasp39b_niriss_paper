import math
import numpy as np
import scipy as sp
from array import *
from scipy import interpolate
from scipy import special
from scipy import interp
import pdb
import scipy
import datetime
from numba import jit
from pickle import *
from scipy.integrate import simps
import numpy
import h5py
import glob


'''******************************KCOEFF_INTERP***************************
Purpose: bilinear interpolates opacities from their native T-P grid to
        given atmosphere T-P profile at each layer

Inputs:
    logPgrid--log Pressure grid of opacities (18 points, 1E-6 - 300 bar)
    logTgrid--log Temperature grid of opacities (43 points, 260 - 4000 K)
    logPatm -- log model atmosphere pressure
    wnogrid -- cross-section/CK table wavenumber grid
    kcoeff -- giant kcoeff array (that comes from xsects function)

Returns:
    kcoeff_interp -- model layer interplated kcoeff's


'''
@jit(nopython=True)
def kcoeff_interp(logPgrid, logTgrid, logPatm, logTatm, wnogrid, kcoeff):

	Ng, NP, NT, Nwno, Nord=kcoeff.shape
	Natm=len(logTatm)
	kcoeff_int=np.zeros((Natm,Nwno,Ng,Nord))

	for i in range(Natm):  #looping through atmospheric layers

		y=logPatm[i]
		x=logTatm[i]

		p_ind_hi=np.where(logPgrid>=y)[0][0]
		p_ind_low=np.where(logPgrid<y)[0][-1]
		T_ind_hi=np.where(logTgrid>=x)[0][0]
		T_ind_low=np.where(logTgrid<x)[0][-1]

		y2=logPgrid[p_ind_hi]
		y1=logPgrid[p_ind_low]
		x2=logTgrid[T_ind_hi]
		x1=logTgrid[T_ind_low]

		for j in range(Ng): #looping through gases
			for k in range(Nwno): #looping through wavenumber
				for l in range(Nord): #looping through g-ord
					#'''
					arr=kcoeff[j,:,:,k,l]
					Q11=arr[p_ind_low,T_ind_low]
					Q12=arr[p_ind_hi,T_ind_low]
					Q22=arr[p_ind_hi,T_ind_hi]
					Q21=arr[p_ind_low,T_ind_hi]
					fxy1=(x2-x)/(x2-x1)*Q11+(x-x1)/(x2-x1)*Q21
					fxy2=(x2-x)/(x2-x1)*Q12+(x-x1)/(x2-x1)*Q22
					fxy=(y2-y)/(y2-y1)*fxy1 + (y-y1)/(y2-y1)*fxy2
					kcoeff_int[i,k,j,l]=fxy
					#'''
	return kcoeff_int


'''******************************XSECTS****************************
Purpose: hacky way of loading in all the cross-sections/CK tables

Inputs:
    -none--just loads in a bunch of H5 files
Returns:
        P--Pressure grid of opacities (18 points, 1E-6 - 300 bar)
        T--Temperature grid of opacities (43 points, 260 - 4000 K)
        wno--wavenumber array for opacities--this sets model grid resolution (here R=250, 100000-100 cm-1)
        g--correlated-K g-ordiances (10 points)
        wts--corrosponding CK weights (10 points)
        xsecarr--contains the CK coefficents (m2/atom)--Ngas x Npressure x Ntemperature x Nwno x Nwts
'''
def xsects(wnomin, wnomax):



    ### Read in CK arrays
    xsecpath='/data/mrline2/ABSCOEFF/CK_COEFFS/R3000_NIRISS/'
    ##continuum----------------------------------------------------
    # H2H2
    file=xsecpath+'H2H2_RICHARD12_CK_R3000_10double_3450_20000wno.h5'
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    wno=np.array(hf['wno'])
    T=np.array(hf['T'])
    P=np.array(hf['P'])
    g=np.array(hf['g'])
    wts=np.array(hf['wts'])
    hf.close()
    xsecarrH2H2=10**(kcoeff-4.)
    # H2He
    file=xsecpath+'H2He_RICHARD12_CK_R3000_10double_3450_20000wno.h5'
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrH2He=10**(kcoeff-4.)

    #normal molecules---------------------------------------------------
    # H2O
    #file='./ABSCOEFF_CK/H2O_CK_R500_10double_750_30000wno.h5'  #OLD
    file=xsecpath+'H2O_POKOZATEL_EHSAN_CK_R3000_10double_3450_20000wno.h5'  #NEW
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrH2O=10**(kcoeff-4.)
    #freedman (PS) H2O - H2He for T <500K
    file=xsecpath+'H2O_PS_FREEDMAN_CK_R3000_10double_3450_20000wno.h5'  #OLD
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrH2O[0:16,]=10**(kcoeff[0:16,]-4.)
    # CO
    file=xsecpath+'CO_HITEMP_HELIOS_CK_R3000_10double_3450_20000wno.h5'
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrCO=10**(kcoeff-4.)
    # CO2
    file=xsecpath+'CO2_FREEDMAN_CK_R3000_10double_3450_20000wno.h5'  #get rid of?
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrCO2=10**(kcoeff-4.)
    # CH4
    file=xsecpath+'CH4_HITEMP_HELIOS_CK_R3000_10double_3450_20000wno.h5'
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrCH4=10**(kcoeff-4.)
    # NH3
    # file=xsecpath+'NH3_EXOMOL_SUPER_HELIOS_CK_R3000_10double_3450_20000wno.h5'  #get rid of?
    # hf=h5py.File(file, 'r')
    # kcoeff=np.array(hf['kcoeff'])
    # hf.close()
    # xsecarrNH3=10.**(kcoeff-4.)
    # H2S
    file=xsecpath+'H2S_EXOMOL_SUPER_HELIOS_CK_R3000_10double_3450_20000wno.h5'  #get rid of?
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrH2S=10**(kcoeff-4.)
    # PH3
    # file=xsecpath+'PH3_FREEDMAN_CK_R3000_10double_3450_20000wno.h5'
    # hf=h5py.File(file, 'r')
    # kcoeff=np.array(hf['kcoeff'])
    # hf.close()
    # xsecarrPH3=10**(kcoeff-4.)
    # HCN
    file=xsecpath+'HCN_EXOMOL_HELIOS_CK_R3000_10double_3450_20000wno.h5'
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrHCN=10**(kcoeff-4.)
    #C2H2
    # file=xsecpath+'C2H2_EXOMOL_SUPER_HELIOS_CK_R3000_10double_3450_20000wno.h5'
    # hf=h5py.File(file, 'r')
    # kcoeff=np.array(hf['kcoeff'])
    # hf.close()
    # xsecarrC2H2=10**(kcoeff-4.)
    #OH
    # file=xsecpath+'OH_HITEMP_HELIOS_CK_R3000_10double_3450_20000wno.h5'
    # hf=h5py.File(file, 'r')
    # kcoeff=np.array(hf['kcoeff'])
    # hf.close()
    # xsecarrOH=10**(kcoeff-4.)

    #Neutral Atomic---------------------------------------------------
    # Na
    file=xsecpath+'Na_FREEDMAN_CK_R3000_10double_3450_20000wno.h5'
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrNa=10**(kcoeff-4.)
    # K
    file=xsecpath+'K_FREEDMAN_CK_R3000_10double_3450_20000wno.h5'
    hf=h5py.File(file, 'r')
    kcoeff=np.array(hf['kcoeff'])
    hf.close()
    xsecarrK=10**(kcoeff-4.)


    #loading mie coefficients-----------------------------
    #loading mie coefficients-----------------------------
    cond_name='MgSiO3'
    file=xsecpath+cond_name+'_r_0.01_300um_wl_0.3_200um_interp_R3000_10double_3450_20000wno.pic'
    wno,radius,Mies = load(open(file,'rb'),encoding='latin1')  #radius is in mixrons!
    SSA=Mies[1,:,:]/Mies[0,:,:]#single scatter albedo
    Mies[1,:,:]=SSA  #single scatter albedo
    Mg2SiO4=Mies  #Mies = Qext, Qs, asym
    xxsec=Mg2SiO4[0,:,:].T*np.pi*radius**2*1E-12 #scattering cross-section
    Mg2SiO4[0,:,:]=xxsec.T


    #mies_arr=np.array([Na2S*0.,KCl*0.,Mg2SiO4*0.,MgSiO3*1.,Al2O3*0.,CaTiO3*0.,Fe*0.])*1.
    mies_arr=np.array([Mg2SiO4*1.])*1.



    xsecarr = np.log10(np.array([xsecarrH2H2, xsecarrH2He,xsecarrH2O,xsecarrCO,xsecarrCO2,xsecarrCH4,xsecarrH2S,xsecarrHCN,xsecarrNa,xsecarrK]))

    wnoloc=np.where((wno >= wnomin) & (wno <= wnomax))[0]
    return P,T,wno[wnoloc],g,wts,xsecarr[:,:,:,wnoloc],radius*1E-6,mies_arr[:,:,:,wnoloc]  #cropping wavenumber grid here



'''******************************CALKTAUXSECCK****************************
Purpose: computes tangent limb transmittances (t(z)) for use in computing transmission spectrum

Inputs:
    -kcoeffs -- CK cross-sections
    -Z -- hydrostatic altitude grid
    -Pavg -- layer midpoint presssures
    -Tavg -- layer midpoint temperatures
    -Fractions -- array containing mixing ratios of each gas vs. layer mp pressures
    -r0 -- reference radius
    -wts -- CK weights
    -Fractions_Continuum -- continuum absorber mixing ratios
    -xsecContinuum -- continuum absorber opacities in hacky CK format
Returns:
    trans -- an array of limb transmittances

'''
@jit
def CalcTauXsecCK(kcoeffs,Z,Pavg,Tavg, Fractions, r0,gord, wts, Fractions_Continuum, xsecContinuum):
    #looks similar to tau-rex doesn't it? Ingo and I (and Marco Rocchetto) co-wrote these CK versions many moons ago.
    #after realizing doing hacky sampling and cross-section averaging don't work well...(beer happned)
    ngas=Fractions.shape[0]
    nlevels=len(Z)
    nwno=kcoeffs.shape[1]
    trans=np.zeros((nwno, nlevels))+1.
    dlarr=np.zeros((nlevels,nlevels))
    ncont=xsecContinuum.shape[-1]
    #pre-comuting path length segemnts along each beam for
    #each tangent height...faster to do outside of wno loop
    for i in range(nlevels-2):
        for j in range(i):
            r1=r0+Z[i]
            r2=r0+Z[i-j]
            r3=r0+Z[i-j-1]
            dlarr[i,j]=np.sqrt(r3**2-r1**2)-np.sqrt(r2**2-r1**2)

    kb=1.38E-23
    #big loop over wavenumber
    for v in range(nwno):
        for i in range(nlevels-2): #loop over tangent heights
            transfull=1.
            #for CK gases--try to do ALL gases as CK b/c of common interpolation
            for k in range(ngas): #loop over gases
                transtmp=0.
                for l in range(len(wts)):
                    tautmp=0.
                    for j in range(i):
                        dl=dlarr[i,j]
                        curlevel=i-j-1
                        u=dl*1E5*Pavg[curlevel]/(kb*Tavg[curlevel])
                        tautmp+=2.*Fractions[k,curlevel]*kcoeffs[curlevel,v,k,l]*u
                    transtmp+=np.exp(-tautmp)*wts[l]/2.
                transfull*=transtmp
            #for continuum aborbers (gas rayligh, condensate scattering etc.--nlayers x nwno x ncont
            #'''
            for k in range(ncont):
                tautmp=0.
                for j in range(i):
                    dl=dlarr[i,j]
                    curlevel=i-j-1
                    u=dl*1E5*Pavg[curlevel]/(kb*Tavg[curlevel])*Fractions_Continuum[k,curlevel]
                    #print(Fractions_Continuum[k,curlevel])
                    tautmp+=2.*xsecContinuum[curlevel,v,k]*u
                transfull*=np.exp(-tautmp)
            #'''
            trans[v,i]=transfull
    return trans


'''******************************TRAN****************************
Purpose: Generates "on-the-fly" transmission spectrum given atmospheric structure

Inputs:
    -xsecs--tuple that contains correlated-K opaccity info
        Pgrid, Tgrid, wno, gord, wts, xsecarr=xsecs
        Pgrid--Pressure grid of opacities (18 points, 1E-6 - 300 bar)
        Tgrid--Temperature grid of opacities (43 points, 260 - 4000 K)
        wno--wavenumber array for opacities--this sets model grid resolution (here R=250, 100000-100 cm-1)
        gord--correlated-K g-ordiances (10 points)
        wts--corrosponding CK weights (10 points)
        xsecarr--contains the CK coefficents (m2/atom)--Ngas x Npressure x Ntemperature x Nwno x Nwts
    T--model Temperature profile
    P-model Pressure grid (T is on P)
    -f_i -- gas volume mixing ratio arrays with pressure (one less than pressure as they are defined at layer mp)
    -mmw -- mean molecular weight array
    -Ps -- reference pressure at which planetary radius is defined
    -amp -- rayleigh scattering amplitude (scaling relative to H2 Ray at 0.43 um--Lecavelier des Etangs et al. 2008 method)
    -power -- scattering power index (e.g., 4 is rayleigh)
    -M -- planet mass (for non-uniform gravity) (in Mj)
    -Rstar -- stellar radius (in Rsun)
    -Rp -- planet radius (in Rjup)

Returns:
    -wno -- model wavenumber grid (same as xsecs) cm-1
    -F -- model transit depth on model wavenumber grid  -- (Rp/Rs)**2
    -Z -- hydrostatic altitude grid (in meters)

'''
def tran(xsecs,T, P,fH2,fHe,fH2O,fCO,fCO2,fCH4,fH2S,fHCN,fNa,fK, mmw,Ps,CldOpac,Pc,amp,power,f_r,M,Rstar,Rp):
    #print "Starting tran at ", datetime.datetime.now().time()

    #Convert parameters to proper units
    #converting molar mixing ratios in ppm to just mole fraction

    #pdb.set_trace()
    #Na and K are fixed in this model but can be made free parameter if desired
    #If T < 800 K set these equal to 0!!!--they condense out below this temperature (roughly)
                            #H2H2,  H2He,   FF,  BF, H2O, CO, CO2, CH4, NH3, H2S, PH3, HCN, C2H2, TiO, VO, SiO, FeH, CaH,MgH,  CrH,ALH,  Na,K,  Fe,Mg,Ca,  C,  Si,Ti,  O,FeII,  MgII, TiII, CaII, CII,H2_UV
    Fractions = np.array([fH2*fH2,fHe*fH2,fH2O,fCO,fCO2,fCH4,fH2S,fHCN,fNa,fK])
                        #H2Ray, HeRay  Ray General,
    Frac_Cont = np.array([fH2,fHe,fH2*0.+1.,fH2*0.+1])  #continuum mole fraction profiles
    Frac_Cont=np.concatenate((Frac_Cont, f_r),axis=0)


    #Load measured cross-sectional values and their corresponding
    #T,P,and wno grids on which they were measured

    Pgrid, Tgrid, wno, gord, wts, xsecarr,radius, Mies=xsecs

    #Calculate Temperature, Pressure and Height grids on which
    #transmissivity will be computed
    n = len(P)
    nv = len(wno)


    Z=np.zeros(n)  #level altitudes
    dZ=np.zeros(n)  #layer thickness array
    r0=Rp*71492.*1.E3  #converting planet radius to meters
    mmw=mmw*1.660539E-27  #converting mmw to Kg
    kb=1.38E-23
    G=6.67384E-11
    M=M*1.89852E27


    #Compute avg Temperature at each grid
    Tavg = np.array([0.0]*(n-1))
    Pavg = np.array([0.0]*(n-1))
    for z in range(n-1):
        Pavg[z] = np.sqrt(P[z]*P[z+1])
        Tavg[z] = interp(np.log10(Pavg[z]),sp.log10(P),T)
    #create hydrostatic altitutde grid from P and T
    Phigh=P.compress((P>Ps).flat)  #deeper than reference pressure
    Plow=P.compress((P<=Ps).flat)   #shallower than reference pressure
    for i in range(Phigh.shape[0]):  #looping over levels above ref pressure
        i=i+Plow.shape[0]-1
        g=G*M/(r0+Z[i])**2#g0*(Rp/(Rp+Z[i]/(69911.*1E3)))**2
        H=kb*Tavg[i]/(mmw[i]*g)  #scale height
        dZ[i]=H*np.log(P[i+1]/P[i]) #layer thickness, dZ is negative
        Z[i+1]=Z[i]-dZ[i]   #level altitude
	#print P[i], H/1000.
    for i in range(Plow.shape[0]-1):  #looping over levels below ref pressure
        i=Plow.shape[0]-i-1
        g=G*M/(r0+Z[i])**2#g0*(Rp/(Rp+Z[i]/(69911.*1E3)))**2
        H=kb*Tavg[i]/(mmw[i]*g)
        dZ[i]=H*np.log(P[i+1]/P[i])
        Z[i-1]=Z[i]+dZ[i]
	#print P[i], H/1000.

    #Interpolate values of measured cross-sections at their respective
    #temperatures pressures to the temperature and pressure of the
    #levels on which the optical depth will be computed
    #print "Interpolating cross-sections at ", datetime.datetime.now().time()
    #make sure   200 <T <4000 otherwise off cross section grid
    TT=np.zeros(len(Tavg))
    TT[:]=Tavg
    TT[Tavg < 500] = 500.
    TT[Tavg > 4000] = 4000.
    PP=np.zeros(len(Pavg))
    PP[:]=Pavg
    PP[Pavg < 3E-6]=3E-6
    PP[Pavg >=300 ]=300

    kcoeffs_interp=10**kcoeff_interp(np.log10(Pgrid), np.log10(Tgrid), np.log10(PP), np.log10(TT), wno, xsecarr)

    #continuum opacities (nlayers x nwnobins x ncont)***********
    xsec_cont=kcoeffs_interp[:,:,0,0]
    wave = (1/wno)*1E8
    sigmaH2 = xsec_cont*0.+1*((8.14E-13)*(wave**(-4.))*(1+(1.572E6)*(wave**(-2.))+(1.981E12)*(wave**(-4.))))*1E-4  #H2 gas Ray
    sigmaHe = xsec_cont*0.+1*((5.484E-14)*(wave**(-4.))*(1+(2.44E5)*(wave**(-2.))))*1E-4   #He gas Ray
    sigmaH = xsec_cont*0.+1.*(5.799e-13/wave**4 + 1.422e-6/wave**6 + 2.784/wave**8)*1E-4 #H gas
    #Rayleigh Haze from des Etangs 2008
    wno0=1E4/0.43
    sigmaRay=xsec_cont*0.+2.E-27*amp*(wno/wno0)**power*1E-4
    #grey cloud opacity
    sigmaCld=xsec_cont*0.+CldOpac

    #mie scattering
    xsecMie=Mies[0,0,:,:].T
    sigmaMie=np.repeat(xsecMie[np.newaxis,:,:],len(Pavg),axis=0)

    xsecContinuum=np.array([sigmaH2.T,sigmaHe.T,sigmaRay.T,sigmaCld.T]).T #building continuum xsec array (same order as cont_fracs)
    xsecContinuum=np.concatenate((xsecContinuum, sigmaMie),axis=2)


    #(add more continuum opacities here and in fractions)
    #********************************************
    #Calculate transmissivity as a function of
    #wavenumber and height in the atmosphere
    #print "Computing Transmittance ", datetime.datetime.now().time()
    t=CalcTauXsecCK(kcoeffs_interp,Z,Pavg,Tavg, Fractions, r0,gord,wts,Frac_Cont,xsecContinuum)
    loc=np.where(Pavg >= Pc)[0]
    t[:,loc]=0.

    '''
    #plot limb transmittance--jankey transmission contribution functions....
    pdb.set_trace()
    fig1, ax=subplots()
    contourf(1E4/wno, P, t.T,levels=np.arange(0.,1.01,0.01))
    xlabel('$\lambda$ ($\mu$m)',size='xx-large')
    ylabel('Pressure [bar]',fontsize=20)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,15, 20])
    ax.axis([0.15,2,200,1E-7])
    minorticks_on()
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.tick_params(length=10,width=1,labelsize='large',which='major')
    savefig('transmittance_H2Hebrd.pdf',fmt='[df')
    show()
    close()
    pdb.set_trace()
    '''

    #Compute Integral to get (Rp/Rstar)^2 (equation in brown 2001, or tinetti 2012)
    F=((r0+np.min(Z))/(Rstar*6.955E8))**2+2./(Rstar*6.955E8)**2.*np.dot((1.-t),(r0+Z)*dZ)
    #print "Ending Tran at ", datetime.datetime.now().time()

    return wno, F, Z




##############################   CLOUD ROUTINES  ###################################
####################################################################################
####################################################################################


'''
#**************************************************************************
FILE:

DESCRIPTION:

INPUTS:

RETURNS:

#**************************************************************************
'''
@jit(nopython=True)
def cloud_profile(fsed,cloud_VMR, P, Pbase):
    Pavg=0.5*(P[1:]+P[:-1])
    cond=cloud_VMR
    loc0=np.where(Pbase >= Pavg)[0][-1]
    cond_mix=np.zeros(len(Pavg))+1E-50
    cond_mix[0:loc0+1]=cond*(Pavg[0:loc0+1]/Pavg[loc0])**fsed  #A&M2001 eq 7., but in P-coordinates (using hydrostatic) and definition of f_sed
    return cond_mix

'''
#**************************************************************************
FILE:

DESCRIPTION:

INPUTS:

RETURNS:

#**************************************************************************
'''
#@jit(nopython=True)
def particle_radius(fsed,Kzz,mmw,T, P,g, rho_c,mmw_c, qc,rr):
    Tavg=0.5*(T[1:]+T[:-1])
    Pavg=0.5*(P[1:]+P[:-1])
    dlnr=np.abs(np.log(rr[1])-np.log(rr[0]))
    kb=1.38E-23  #boltzman constant
    mu0=1.66E-27  #a.m.u.
    d=2.827E-10  #bath gas molecule diameter (m)
    alpha=1.4  #alpha factor from A&M 2001 (don't need to change this)
    sig_eff=2  #log-normal particle size distribution width

    #atmosphere properties
    H=kb*Tavg/(mmw*mu0*g)  #scale height
    rho_a=Pavg*mmw*mu0*1E5/(kb*Tavg)  #atmospheric mass density

    wmix=Kzz/H  #vertical mixing velocity
    mfp=kb*Tavg/(2**0.5*np.pi*d**2*Pavg*1E5)   #mean free path
    eta=5./16.*np.sqrt(np.pi*2.3*mu0*kb*Tavg)*(Tavg/59.7)**.16/(1.22*np.pi*d**2) #dynamic viscosity of bath gas

    #computing varius radius profiles
    r_sed=2./3.*mfp*((1.+10.125*eta*wmix*fsed/(g*(rho_c-rho_a)*mfp**2))**.5-1.)  #sedimentation radius
    r_eff=r_sed*np.exp(-0.5*(alpha+1)*np.log(sig_eff)**2)  #A&M2011 equation 17 effective radius
    r_g=r_sed*np.exp(-0.5*(alpha+6.)*np.log(sig_eff)**2) #A&M formula (13)--lognormal mean (USE THIS FOR RAD)

    #droplet VMR
    f_drop=3.*mmw_c*mu0*qc/(4.*np.pi*rho_c*r_g**3)*np.exp(-4.5*np.log(sig_eff)**2)  #
    prob_lnr=np.zeros((len(rr),len(r_g)))
    for i in range(len(prob_lnr)): prob_lnr[i,:]=1./((2.*np.pi)**0.5*np.log(sig_eff))*np.exp(-0.5*np.log(rr[i]/r_g)**2/np.log(sig_eff)**2)*dlnr
    f_r=prob_lnr*f_drop
    return r_sed, r_eff, r_g, f_r



'''******************************INSTRUMENT_BIN****************************
Purpose: Bins model spectrum to instrumental wavelength grid
        assuming a simple "top-hat" IP (e.g., "averaging")

Inputs:
    -wlgrid--data wavelength grid (in microns)
    -bin_wid--data wavelength grid bin widdths (full widths)
    -wno--model wavenumber grid (sorry observers---us modeler people use wavneumbers)
    -Depth--model transit depth (Rp/Rs)**2.

Returns:
    -Depth_binned--transit depth binned to data wlgrid

'''
def instrument_bin(wlgrid,bin_wid,wno, Depth):
    delta=bin_wid
    Nbin=len(wlgrid)
    Depth_binned=np.zeros(Nbin)
    for i in range(1,Nbin):
        loc=np.where((1E4/wno > wlgrid[i]-0.5*delta[i-1]) & (1E4/wno < wlgrid[i]+0.5*delta[i]))
        Depth_binned[i]=np.mean(Depth[loc])

    i=0
    loc0=np.where((1E4/wno > wlgrid[i]-0.5*delta[i]) & (1E4/wno < wlgrid[i]+0.5*delta[i]))
    Depth_binned[i]=np.mean(Depth[loc0])
    return Depth_binned


'''******************************FIND_NEAREST****************************
Purpose: Finds the nearest value in an array
Inputs:
    -array--array to search
    -value--value to find nearest for
Returns:
    -array[idx]--nearest value to "value" in array

'''
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

'''******************************RETURN_FILE****************************
Purpose: takes in grid model params and returns the appropriately named file.
        if an "off grid" point is specified, a nearest neighbor routine will
         be used to find the closest point.  Right now, the notebook and model
        grid are perfectly aligned. If you modifiy, you will have to be aware of this
Inputs:
    -redist--model redistribution factor (here defined to be (Tirr/Teq)**4, where 1=full, 2 = dayside, 2.66 = max)
    -logMet--log metallicity relative to solar, e.g., [M/H]. [M/H]=0, 1, 2 = 1x, 10x, 100x solar etc
    -CtoO--carbon-to-oxygen ratio. Solar is 0.55
    -logKzz--log of the eddy diffusivity in cgs. In these grids, a cconst w/ alt Kzz is used and
             the Zahnle & Marley 2016 time scale prescription to "quench" NH3 and CO, H2O, CH4.
Returns:
    -filearr--filename of corrosponding model grid.

Notes:
    -this whole function can be modified to read in another model grid. Just make sure you mod
    the below grid parameters accordingly
'''
def return_file(redist, logMet, CtoO, logKzz):

    #redist
    arr=np.array([0.657, 0.721, 0.791, 0.865, 1.   , 1.03 , 1.12 , 1.217, 1.319])
    red=find_nearest(arr,redist)

    #met
    arr=np.array([-1., -0.875, -0.75, -0.625, -0.5, -0.375,-0.25,-0.125, 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.375, 2.5])
    logZ=find_nearest(arr,logMet)

    #CtoO
    arr=np.array([0.1, 0.2, 0.35, 0.45, 0.55, 0.65, 0.7, 0.725, 0.75, 0.775, 0.8, 0.85, 0.9, 0.95])
    CtO=find_nearest(arr,CtoO)

    #logKzz
    arr=np.array([-1.0])
    Kz=find_nearest(arr,logKzz)

    posneg=''
    if logZ >=0.0: posneg='+'
    directory_grid_mrline='/data/mrline2/MRLINE/EXOPLANET_TRANSITS/IRRADIATED_PLANETS/INDIVIDUAL_PLANETS_NEW/WASP-39b_ERS_R50/'
    filearr=sorted(glob.glob(directory_grid_mrline+'OUTPUT/SLIDER_PIC/redist_'+str(red)+'_logZ_'+posneg+str(logZ)+'_CtoO_'+str(CtO)+'_logKzz_'+str(Kz)+'_Teff_150.0.pic'))
    return filearr[0]

def return_ko_file(redist, logMet, CtoO, logKzz, logxKtoO):
    #higher density--done
    # redist=np.array([0.657,0.721, 0.791,0.865, 1,1.03 , 1.12 , 1.217, 1.319])
    # logMet=np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,1.875,2.0,2.125,2.25])
    # CtoO=np.array([0.2, 0.35, 0.45, 0.55, 0.65, 0.7, 0.8])
    # logxKtoO=np.array([-1,-0.5,0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9,1.0 ])

    # print(redist, logMet, CtoO,logxKtoO)
    #redist
    arr=np.array([0.657, 0.721, 0.791, 0.865, 1.0 , 1.03 , 1.12 , 1.217, 1.319])
    red=find_nearest(arr,redist)

    #met
    arr=np.array([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75,1.875,2.0,2.125,2.25])
    logZ=find_nearest(arr,logMet)

    #CtoO
    arr=np.array([0.2, 0.35, 0.45,0.55, 0.65, 0.7, 0.8])
    CtO=find_nearest(arr,CtoO)

    #logKzz
    arr=np.array([-1.0])
    Kz=find_nearest(arr,logKzz)

    arr=np.array([-1,-0.5,0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9,1.0])
    xlogKO=find_nearest(arr,logxKtoO)

    ko_sign=''
    if logxKtoO>=0.0: ko_sign='+'

    if xlogKO==0.0: ko_sign='+'


    posneg=''
    if logZ >=0.0: posneg='+'

    # print(str(red),str(logZ),str(CtO),ko_sign, str(xlogKO))
    directory_grid_mrline='/data/mrline2/MRLINE/EXOPLANET_TRANSITS/IRRADIATED_PLANETS/INDIVIDUAL_PLANETS_NEW/WASP-39b_ERS_R50_KtoO/'
    filearr=sorted(glob.glob(directory_grid_mrline+'OUTPUT/SLIDER_PIC_03/redist_'+str(red)+'_logZ_'+posneg+str(logZ)+'_CtoO_'+str(CtO)+'_xxKtoO_'+ko_sign+str(xlogKO)+'_logKzz_'+str(Kz)+'_Teff_150.0.pic'))
    try:
        return filearr[0]
    except IndexError:
        print('The missing point is ')
        print(red,logZ,CtO,xlogKO)
        filearr=sorted(glob.glob(directory_grid_mrline+'OUTPUT/SLIDER_PIC_03/redist_1.319_logZ_+2.25_CtoO_0.8_xxKtoO_+1.0_logKzz_-1.0_Teff_150.0.pic'))
        return filearr[0]
