#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 10:55:36 2022

Python code -- original and modified versions of Noah MYJ and Noah MP
Reference:
Maillard, J. et al, Evaluation of surface layer scheme representation of temperature inversions over boreal forests in WRF 3.9.1.1,
Geophysical Model Development, 2022, submitted

@author: maillardj
"""

import numpy as np

#%% CONSTANTS AND STABILITY FUNCTION DEFINITION
NU      = 1.47e-5   # m2 s-1
CP      = 1005
RHO     = 1.4       # kg m-3
RD      = 287.5
KAPPA   = 0.4
GRAV    = 9.81      # m s-2
SIGMA   = 5.67e-8

def psiWRF(z_L):    # WRF stability function, as used in the MYJ surface layer scheme
    if z_L>=0.:
        a = -(0.7*z_L + 0.75*z_L*(6-0.35*z_L)*np.exp(-0.35*z_L))
    else:
        X = np.sqrt(np.sqrt(1.-16.*z_L))
        a=2.*np.log((X*X+1.)/2.)
    return a

def psiBUSINGER(z_L):   # Businger-Dyer stability function
    alpha = 4.7
    if z_L >= 0:
        psi = -alpha*z_L
    else:
        X = np.sqrt(np.sqrt(1.-16.*z_L))
        psi=2.*np.log((X*X+1.)/2.)
    return psi

def psiPRR(z_L):    # Poker Flats Research Range stability function (Maillard, 2022)
    a = -5
    b = 0.1
    c = 20
    if z_L>=0.:
        psi = a*z_L**(-3/4*np.arctan(c*z_L-b)*2/np.pi+1.25)
    else:
        X = np.sqrt(np.sqrt(1.-16.*z_L))
        psi=2.*np.log((X*X+1.)/2.)
    return psi

def psiPAULSONm(XX):
    return -2.0* np.log ( (XX +1.0)*0.5) - np.log ( (XX * XX +1.0)*0.5) +2.0* np.arctan(XX) -3.14159265 / 2.0

def psiPAULSONh(XX):
    return  -2.0* np.log( (XX * XX +1.0)*0.5)

#%% NOAH - MYJ SURFACE LAYER

'''
Code for the Noah - MYJ combination. Input parameter:
- version:  ORIG = original (WRF 3.9.1.1) (default)
            MODIF = with modifications described in  Maillard (2022)
- z0m: momentum roughness length of the surface
            0.4 m (default)
- za: height of surface layer top
            15 m (default)
- d: displacement height due to the vegetation
            0 m (default)
'''
class noahmyj():
    LAMBS = 0.3
    EPSILONS = 0.99
    CZIL = 0.1
    
    def __init__(self, version = 'ORIG', z0m = 0.4, za = 15, d = 0):
        self.version    = version
        self.z0m        = z0m
        self.za         = za
        self.d          = d
        
    def solve(self, Ua, Ta, LWd, dsnow, Tg, niter = 10):
        # INPUT VARIABLES
        # Ua:       wind speed at top of the surface layer (za), in m s-1
        # Ta:       temperature at top of the surface layer (za), in K
        # LWd:      downwards longwave flux at the top of the surface layer, in W m-2
        # dsnow:    snow depth, in m
        # Tg:       ground temperature below the snow, in K
        #
        # OUTPUT VARIABLE
        # Ts:       surface temperature in K
        
        RLambs = self.LAMBS/dsnow
        
        # initialize variables
        Ts = Ta-1.
        CH = 0.01
        ustar = 0.3
        
        # loop while calculating turbulent fluxes and surface temperature
        for n in range(niter):
            # calculate turbulent parameter
            CH, ustar   = self.myjsurface(Ta,Ts,Ua,CH,ustar)
            
            # deduce surface temperature
            Ts          = self.noahlsm(Ta, CH, LWd, RLambs, Tg)
        
        return Ts
        
    def myjsurface(self, Ta, Ts, Ua, CHinit, ustarinit):
        z_pbl   = 1000. 
        Czetmax = 10.
        Ric     = 0.505
        
        if self.version == 'ORIG':
            psi     = psiWRF
            zetamax = 1.
        elif self.version == 'MODIF':
            psi     = psiPRR
            zetamax = 100.
        
        deltaT  = (Ta-Ts)
        Ua      = max(Ua,1e-3)
        Rib     = (GRAV/Ta) * deltaT * (self.za-self.d) / Ua**2
        if deltaT>0.:
            Zzil    = self.CZIL*(1+Czetmax*(Rib/Ric)**2) if Rib<=Ric else self.CZIL*(1+Czetmax)
        else:
            Zzil    = self.CZIL
                
        ustar   = ustarinit
        CH      = CHinit
        
        # iteration to determine turbulent coefficients
        for n in range(5):    
            
            # calculate heat roughness length
            z0h     = max(self.z0m * np.exp(-KAPPA*Zzil*np.sqrt(ustar*self.z0m/NU)),1e-28)
            
            # calculate inverse Monin-Obukhov length
            invLMO  = KAPPA * GRAV/Ta * CH * deltaT / ustar**3
            
            # calculate zeta
            zetal   = min((self.za-self.d)*invLMO,  zetamax)
            zetash  = min(z0h*invLMO,               zetamax/(self.za-self.d)*z0h)
            zetasm  = min(self.z0m*invLMO,          zetamax/(self.za-self.d)*self.z0m)
            
            # calculate CH and AKMS
            CH      = max(ustar * KAPPA / (np.log((self.za-self.d)/z0h) - psi(zetal) + psi(zetash)),        0.0001/self.za)
            AKMS    = max(ustar * KAPPA / (np.log((self.za-self.d)/self.z0m) - psi(zetal) + psi(zetasm)),   0.0001/self.za)
            
            # Beljaars correction to ustar
            if deltaT<=0 :
                correc  = 1.2 * abs(GRAV/Ta * z_pbl * CH * deltaT)**(2/3)
            else:
                correc  = 0.
            Ua_cor  = np.sqrt(Ua**2 + correc**2)
            
            # calculate ustar
            ustar   = max(np.sqrt(AKMS*Ua_cor),1e-9)
        
        return CH, ustar
    
    def noahlsm(self, Ta, CH, LWd, RLambs, Tg):
        RCH = RHO * CP * CH
        r   = (4*self.EPSILONS*SIGMA*Ta**3) / RCH
        
        YY  =  Ta + ((LWd-self.EPSILONS*SIGMA*Ta**4)/RCH)/(r+1)
        ZZ  = RLambs/(RCH*(r+1))
        Ts  = (YY + ZZ*Tg)/(ZZ+1)
        
        return Ts
  
#%% NOAH - MP SURFACE LAYER
'''
Code for Noah - MP. Input parameters:
- version:  ORIG = original (WRF 3.9.1.1) (default)
            MODIF = with modifications described in  Maillard (2022)
- z0m: momentum roughness length of the surface
            0.4 m (default)
- za: height of surface layer top
            15 m (default)
- hcan: canopy height
            3 m (default)
- d: displacement height due to the vegetation
            1.1 m (default)
- lai: leaf area index
            0.73 (default)
- epsilonc: canopy emissivity
            1 - exp(-lai) (default)
- fveg: vegetation fraction
            1 - exp(-0.52 lai) (default)
'''
class noahmp():
    LAMBS       = 0.3       # snow conductivity (in W m-1 K-1)
    EPSILONS    = 0.99      # snow emissivity (unitless)
    CZIL        = 0.1
    Z0MG        = 0.002     # ground momentum roughness length (in m)
    CWP         = 0.18  
    DLEAF       = 0.04      # leaf diameter
    KFAC        = 0.001     # K factor (see Maillard 2022)
    
    def __init__(self, version = 'ORIG', z0m = 0.4, za = 15, d = 1.1, hcan = 3, lai = 0.73,
                 epsilonc = np.nan, fveg = np.nan):
        self.version    = version
        self.z0m        = z0m
        self.za         = za 
        self.hcan       = hcan
        self.d          = d
        self.lai        = lai
        self.epsilonc   = 1-np.exp(-self.lai)       if np.isnan(epsilonc)   else epsilonc
        self.fveg       = 1-np.exp(-0.52*self.lai)  if np.isnan(fveg)       else fveg
   
    def solve(self, Ua, Ta, LWd, dsnow, Tg, niter = 10):
        # INPUT VARIABLES
        # Ua:       wind speed at top of the surface layer (za), in m s-1
        # Ta:       temperature at top of the surface layer (za), in K
        # LWd:      downwards longwave flux at the top of the surface layer, in W m-2
        # dsnow:    snow depth, in m
        # Tg:       ground temperature below the snow, in K
        #
        # OUTPUT VARIABLE
        # Ts:       surface temperature in K
        # Tc:       canopy (air) temperature in K
        
        RLambs  = self.LAMBS/dsnow
        dg = dsnow
        if self.version == 'ORIG':
            Ua  = max(Ua,   1)
        else:
            Ua  = max(Ua,    1e-2)
        # initialize variables
        Ts      = Ta-1.     # surface temperature (average)
        Tveg    = Ta-0.5    # canopy vegetation temperature
        Tc      = Ta-0.5    # canopy air temperature
        Tgv     = Ta-1.     # surface temperature, vegetated fraction
        Tgb     = Ta-1.     # surface temperature, bare fraction
        
        # iterate to solve
        for n in range(niter):
            # vegetated fraction
            if self.fveg > 0:
                Tgv, Tv, Tc   = self.noahmpveg(Ta,Ua,Ts,Tveg,LWd,Tg,RLambs)
            else:
                Tgv, Tv, Tc   = 0., np.nan, np.nan
            
            # bare fraction
            Tgb     = self.noahmpbare(Ta,Ua,Ts,LWd,Tg,RLambs,dg)
            
            # average surface temp
            Ts      = self.fveg*Tgv + (1-self.fveg)*Tgb
            Tveg    = Tv

        return Ts, Tc
            
    def noahmpveg(self,Ta, Ua, Ts, Tveg, LWd, Tg, RLambs):
        # initialize variables
        Tc      = Tveg
        Tgv     = Ts
        
        ustar   = 0.1
        AKMS    = 0.
        CH      = 0.
        invLMO  = 0.
        
        ustarg  = 0.1
        CHg     = 0.
        
        FHG     = 0.
        SHg     = 0.
        SHa     = 0.
        WSTAR2  = 0.
    
        VAIE    = min(6,self.lai/self.fveg)
        UC      = Ua * np.log((self.hcan-self.d+self.z0m)/self.z0m) / np.log(self.za/self.z0m) # !! A CHECKER
        
        AIR     = -self.epsilonc*(1.+(1.-self.epsilonc)*(1.-self.EPSILONS))*LWd - self.epsilonc*self.EPSILONS*SIGMA*Tgv**4
        CIR     = (2.-self.epsilonc*(1.-self.EPSILONS))*self.epsilonc*SIGMA
        
        # Solve canopy and vegetation temperatures
        for ii in range(20):
            if self.version == 'ORIG':
                z0h     = self.z0m
                z0hg    = self.Z0MG
                
                # calculate turbulent diffusion coefficients canopy - air
                CH, AKMS, ustar, invLMO, WSTAR2 = self.mpsurface2(ii,Ta,Tc,Ua,ustar,AKMS,CH,invLMO,
                                                                  WSTAR2,self.za, self.z0m, 0.)
                
                CH2     = CH/Ua
                RAHC    = max(1,1/(CH2*Ua))
                
                # calculate turbulent diffusion coefficients canopy - tree and canopy - ground
                RAHG,RB,FHG                     = self.aerocoeff(ii, SHg, UC, Tc, ustar, z0hg, z0h,
                                                                 FHG,VAIE)
                # solve canopy energy balance
                CAH     = 1./RAHC 
                CVH     = 2.*VAIE/RB
                CGH     = 1./RAHG
                COND    = CAH + CVH + CGH
                ATA     = (Ta*CAH + Tgv*CGH) / COND
                BTA     = CVH/COND
                CSH     = (1.-BTA)*RHO*CP*CVH
                Tc      = ATA + BTA*Tveg        # update canopy air temperature
                
                # solve vegetation energy balance
                IRC     = self.fveg*(AIR + CIR*Tveg**4)
                SHC     = self.fveg*RHO*CP*CVH * (Tveg - Tc)
                B       = -IRC-SHC
                A       = self.fveg*(4.*CIR*Tveg**3 + CSH)
                DTV     = B/A
                
                IRC     = IRC + self.fveg *4.*CIR*Tveg**3*DTV
                SHC     = SHC + self.fveg *CSH*DTV
                Tveg    = Tveg + DTV            # update vegetation temperature
                
                SHg     = RHO*CP*(Tgv   - Tc)/RAHG
                SHa     = RHO*CP*(Tc    - Ta)/RAHC
            
            elif self.version == 'MODIF':
                CH, ustar       = self.myjsurface(Ta,Tc,Ua,CH, ustar,za = self.za,z0m = self.z0m,
                                                  d = self.d,z0h = 1e-2*self.z0m)
                RAHC    = max(1,1/(CH + self.KFAC))
                
                CHg, ustarg     = self.myjsurface(Tc,Tgv,UC,CHg,ustarg,za=1.5,z0m=self.Z0MG,
                                                  d=0.,z0h=1e-2*self.Z0MG)
                RAHG    = 1/(CHg)
                
                SHa     = RHO * CP * (Tc - Ta)/RAHC
                SHG     = RHO * CP * (Tc - Tgv)/RAHG
                IRC     = (AIR + CIR*Tc**4)
                B       = -IRC-SHG-SHa
                A       = (4.*CIR*Tc**3 + RHO*CP/RAHC + RHO*CP/RAHG)
                DTV     = B/A
                Tc      = Tc + DTV
                Tveg    = Tc
            
                SHa     = RHO * CP * (Tc    - Ta)/RAHC
                SHg     = RHO * CP * (Tgv   - Tc)/RAHG
        
        # Solve surface temperature
        AIR     = - self.EPSILONS*(1.-self.epsilonc)*LWd - self.EPSILONS*self.epsilonc*SIGMA*Tveg**4
        CIR     = self.EPSILONS*SIGMA
        CSH     = RHO*CP/RAHG
        CGH     = RLambs
        niter   = 5
        for ii in range(niter):
            IRG     = CIR*Tgv**4 + AIR
    
            SHG     = CSH * (Tgv- Tc)
            GH      = CGH * (Tgv - Tg)
            
            B       = -IRG-SHG-GH
            A       = 4.*CIR*Tgv**3 + CSH + CGH
            DTG     = B/A
            
            Tgv     = Tgv  + DTG
    
        return Tgv, Tveg, Tc
    
    def noahmpbare(self, Ta, Ua, Ts, LWd, Tg, RLambs, dg):   
        Tgb     = Ts
        ustar   = 0.1
        AKMS    = 0.
        CH      = 0.
        invLMO  = 0.
        
        SHB     = 0.
        WSTAR2  = 0.
        
        CIR     = self.EPSILONS * SIGMA
        
        niter   = 5
        for ii in range(niter):
            # calculate turbulent diffusion coeffs
            if self.version == 'ORIG':
                CH, AKMS, ustar, invLMO, WSTAR2     = self.mpsurface2(ii,Ta,Tgb,Ua,ustar,AKMS,CH,invLMO,
                                                                  WSTAR2,self.za, self.Z0MG, 0.)
                
                CH2     = CH/Ua
                CH2     = min(0.01, CH2)
                RAHB    = max(1,    1/(CH2*Ua))
            elif self.version == 'MODIF':
                CH, ustar                           = self.myjsurface(Ta,Tgb,Ua,CH, ustar,za = self.za,
                                                                      z0m = self.Z0MG,d = 0.,z0h = 1e-2*self.Z0MG)
                CH2     = CH/Ua
                RAHB    = max(1,    1/(CH2*Ua + self.KFAC))

            # Solve SEB to get surface temp
            Csh     = RHO*CP/RAHB
        
            #fluxes
            IRB     = CIR*Tgb**4 - self.EPSILONS * LWd
            SHB     = Csh * (Tgb - Ta)
            GHB     = RLambs * (Tgb - Tg)
            
            B       = -IRB - SHB - GHB
            A       = 4*CIR*Tgb**3 + Csh + RLambs
            dTgb    = B/A
    
            Tgb     = Tgb + dTgb
        
        return Tgb
    
    def myjsurface(self, Ta, Ts, Ua, CHinit, ustarinit, za, z0m, d, z0h):
        z_pbl   = 1000. 
        psi     = psiPRR
        zetamax = 1000.
        
        deltaT  = (Ta-Ts)
        Ua      = max(Ua,1e-3)                
        ustar   = ustarinit
        CH      = CHinit
        
        # iteration to determine turbulent coefficients
        for n in range(1):    
            
            # calculate inverse Monin-Obukhov length
            invLMO  = KAPPA * GRAV/Ta * CH * deltaT / ustar**3
            
            # calculate zeta
            zetal   = min((za-d)*invLMO,  zetamax)
            zetash  = min(z0h*invLMO,               zetamax/(za-d)*z0h)
            zetasm  = min(z0m*invLMO,          zetamax/(za-d)*z0m)

            # calculate CH and AKMS
            CH      = max(ustar * KAPPA / (np.log((za-d)/z0h) - psi(zetal) + psi(zetash)),   0.0001/za)
            AKMS    = max(ustar * KAPPA / (np.log((za-d)/z0m) - psi(zetal) + psi(zetasm)),   0.0001/za)

            # Beljaars correction to ustar
            if deltaT<=0 :
                correc  = 1.2 * abs(GRAV/Ta * z_pbl * CH * deltaT)**(2/3)
            else:
                correc  = 0.
            Ua_cor  = np.sqrt(Ua**2 + correc**2)
            
            # calculate ustar
            ustar   = max(np.sqrt(AKMS*Ua_cor),1e-9)
        
        return CH, ustar
    
    def aerocoeff(self, ii, SHg, UC, Tc, ustar, z0hg, z0h, FHG, VAIE):
        invLMO  = 0.
        MOLG    = 0.
        
        if ii > 0.:
            tmp1    = KAPPA * GRAV / Tc * SHg/(RHO*CP)
            if abs(tmp1)<1e-6: 
                tmp1    = 1e-6
            MOLG    = -1 * ustar**3/tmp1
            invLMO  = min((self.d-self.Z0MG)/MOLG,  1.) 
        
        if (invLMO < 0.):
            FHGNEW  = (1. - 15.*invLMO)**(-0.25)
        else:
            FHGNEW  = 1 + 4.7*invLMO
        
        if ii == 0.:
            FHG     = FHGNEW
        else:
            FHG     = (FHGNEW + FHG)/2
            
        CWPC    = (self.CWP * VAIE * self.hcan * FHG)**0.5 
        TMP1    = np.exp(-CWPC*z0hg/self.hcan)
        TMP2    = np.exp(-CWPC*(self.d+z0h)/self.hcan)
        TMPRAH2 = self.hcan * np.exp(CWPC) / CWPC * (TMP1-TMP2)
        KH      = max( KAPPA * ustar * (self.hcan-self.d),  1e-6)
        RAHG    = TMPRAH2/KH 
    
        TMPRB  = CWPC*50. / (1. - np.exp(-CWPC/2.))
        RB     = TMPRB * np.sqrt(self.DLEAF/UC) 

        return RAHG, RB, FHG
        
    def mpsurface2(self, ii, Ta, Tc, Ua, ustar, AKMS, CH, invLMO, WSTAR2, za, z0m, d):
        # initializations
        EXCM    = 0.001
        EPSU2   = 1e-4
        WWST    = 1.2
        ZTMIN   = -5.
        EPSUST  = 0.07
        ZTMAX   = 1.
        
        Zilfc   = - self.CZIL * KAPPA * 1/np.sqrt(NU)
        RDZ     = 1.0/ za
        CXCH    = EXCM * RDZ
        DTHV    = Ta-Tc
        
        DU2     = max(Ua**2,    EPSU2)
        BTGH    = GRAV / 270 * 1000
        
        Ua      = max(Ua,       1e-4)
        
        # calculate friction velocity and inverse Monin-Obukhov length for first iteration
        if ii==0: 
            if (BTGH * CH * DTHV <= 0.0):
               WSTAR2   = WWST**2* abs(BTGH * CH * DTHV)** (2.0/3.0)
            else:
               WSTAR2   = 0.0
            ustar   = max(np.sqrt(AKMS*np.sqrt(DU2+WSTAR2)),    EPSUST)
            invLMO  = KAPPA * GRAV/270 * CH * DTHV/ ustar**3
            
        # calculate various lengths
        z0h     = max(1e-6, z0m * np.exp(Zilfc*np.sqrt(ustar*z0m)))
        ZSLU    = za + z0m
        ZSLT    = za + z0h
        RLOGU   = np.log(ZSLU / z0m)
        RLOGT   = np.log(ZSLT / z0h)
    
        ZETALT  = max(ZSLT * invLMO,    ZTMIN)
        ZETALU  = ZSLU * invLMO
        ZETAU   = z0m * invLMO
        ZETAT   = z0h * invLMO
        
        # calculate stability functions
        if (invLMO < 0.0):
              XLU4  = 1.0 -16.0* ZETALU
              XLT4  = 1.0 -16.0* ZETALT
              XU4   = 1.0 -16.0* ZETAU
              XT4   = 1.0 -16.0* ZETAT
              XLU   = np.sqrt(np.sqrt(XLU4))
              XLT   = np.sqrt(np.sqrt(XLT4))
              XU    = np.sqrt(np.sqrt(XU4))
              XT    = np.sqrt(np.sqrt(XT4))
              PSMZ  = psiPAULSONm(XU)
              SIMM  = psiPAULSONm(XLU) - PSMZ + RLOGU
              PSHZ  = psiPAULSONh(XT)
              SIMH  = psiPAULSONh(XLT) - PSHZ + RLOGT
        else:
              ZETALU= min(ZETALU,   ZTMAX)
              ZETALT= min(ZETALT,   ZTMAX)
              ZETAU = min(ZETAU,    ZTMAX/(ZSLU/z0m))   
              ZETAT = min(ZETAT,    ZTMAX/(ZSLT/z0h))  
              PSMZ  = 5.*ZETAU
              SIMM  = 5.*(ZETALU) - PSMZ + RLOGU
              PSHZ  = 5.*(ZETAT)
              SIMH  = 5.*(ZETALT) - PSHZ + RLOGT
              
        # calculate AKMS and CH
        ustar   = max(np.sqrt(AKMS * np.sqrt(DU2+ WSTAR2)), EPSUST)
        z0h     = max(1.0E-6,                               np.exp(Zilfc * np.sqrt(ustar * z0m))* z0m)
        ZSLT    = za + z0h
        
        RLOGT   = np.log(ZSLT / z0h)
        USTARK  = ustar * KAPPA
        
        if (SIMM < 1.0e-6):
            SIMM    = 1.0e-6
        AKMS    = max(USTARK / SIMM,    CXCH)
    
        if SIMH < 1.0e-6:
            SIMH    = 1.0e-6        
        CH      = max(USTARK / SIMH,    CXCH)
        
        # Beljaars correction
        if (BTGH * CH * DTHV <= 0.0):
           WSTAR2   = WWST**2. * abs(BTGH * CH * DTHV)** (2.0/3.0)
        else:
           WSTAR2   = 0.0
        
        # calculate new inverse Monin-Obukhov length
        RLMN    = KAPPA * GRAV/270 * CH * DTHV / ustar**3
        RLMA    = invLMO * 0.15 + RLMN * 0.85
        invLMO  = RLMA
    
        return CH, AKMS, ustar, invLMO, WSTAR2