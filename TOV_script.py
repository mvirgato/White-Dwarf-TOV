from scipy.interpolate.interpolate import interp1d
from constants import *
from finite_T_FMT import elem_data as elem_data

import numpy as np
import pandas as pd
from scipy import integrate
from matplotlib import pyplot as plt
from glob import glob
from scipy import interpolate as interp
from scipy.stats import gmean
import os

plt.style.use('mvstyle')

def file_filter(elememt, temp, files):
    T = '{:0.3e}'.format(temp)
    file = [x for x in files if elememt in x]
    file = [x for x in file if T in x]
    return file[0]


# eos_files = glob('results/*/**/EoS/FMT_finite_T.dat')
def B_prof_sover(rad, M_dat, P_dat):
    '''
    Input rad in km, M_dat in Msun, P_dat in dyne/cm^2
    '''

    RS = rad[-1]
    x_prof = rad/RS
    M_prof = interp1d(x_prof, M_dat*Msol, bounds_error=False)
    P_prof = interp1d(x_prof, P_dat*0.1, bounds_error=False)  # convert to Pa
    # print(x_prof)
    RS = RS*1e3  # convert to meters

    def dBdr(x, B):

        r = x*RS

        num1 = B*RS
        num2 = (2.0 * GNewt/cspeed**2/r**2)
        num3 = (M_prof(x) + 4.0*pi * P_prof(x) * r**3 / cspeed**2)

        denom = 1.0 - 2.0 * GNewt * M_prof(x)/cspeed**2 / r

        return num1 * num2 * num3 / denom

    def rev_dBdr(x, B):
        return -1.0*dBdr(-x, B)

    x1 = -x_prof[-1]
    x2 = -x_prof[0]

    B_R = 1.0 - 2.0 * GNewt * M_dat[-1]*Msol/RS/cspeed**2
    sol = integrate.solve_ivp(rev_dBdr, t_span=[x1, x2], y0=[ B_R], t_eval=np.flip(-1.0*x_prof),  vectorized=True)

    B_prof = np.array(sol.y)[0]
    B_prof = np.flip(B_prof)

    return B_prof

def surf_gravity(Rstar, Mstar):

    newt_part = GNewt * Mstar/2.0/Rstar**2
    GR_corr = 1.0 - 2.0 * GNewt * Mstar / cspeed**2 / Rstar

    return newt_part*GR_corr

def T_eff(ne, nB, Amass, Tc, r_dat, Mstar):

    '''
    Input Mstar in solar masses
    '''
    
    Rstar = r_dat[-1]
    Ye = ne/nB

    # plt.loglog(r_dat, Ye)
    # plt.loglog(r_dat, ne)
    # plt.loglog(r_dat, nB)
    # plt.show()
    YeInterp = interp1d(r_dat/Rstar, Ye)

    def integrand(x):
        return YeInterp(x)*x**2

    # YeAvg = 3.0 * integrate.quad(integrand, r_dat[0]/Rstar + 1e-4, 1, limit = 100)[0]
    YeAvg = gmean(Ye)
    mue = 1/YeAvg

    const = 5.7e5
    Teff4 = const * Amass * Mstar * Tc**3.5 / ( 4.0 * pi * (sigma_SB * JouleTOerg /cmTOm**2) * Rstar**2 * mue**2 * Msol)
    # print(Amass, Rstar,  Mstar, mue, YeAvg, Tc, Teff4**0.25)

    return Teff4**0.25



class eos_class:

    def __init__(self, element, temperature):

        self.eos_files = glob('results/*/**/EoS/FMT_finite_T.dat')

        self.elem = elem_data(element, float(temperature))
        

        self.eos_name = file_filter(element, temperature, self.eos_files)
        self.eos_data = pd.read_csv(self.eos_name, header = 0, sep = '\t')

        self.xmin = self.elem.xc
        self.xmax = self.eos_data['xWS'].iloc[-1]
        
        self.xws_values = self.eos_data['xWS'][self.eos_data['xWS']>self.elem.xc]
        self.rho_values = self.eos_data['rho[g/cm^3]'][self.eos_data['xWS']>self.elem.xc]
        self.ne_values = self.eos_data['n_e[pm^-3]'][self.eos_data['xWS']>self.elem.xc]
        self.P_values = self.eos_data['P[N/m^2]'][self.eos_data['xWS']>self.elem.xc]
        self.mufe_values = self.eos_data['mu_F_e[MeV]'][self.eos_data['xWS']>self.elem.xc]

        self.delta_x = (self.xws_values.iloc[2] - self.xws_values.iloc[1])/100

        self.P_min = self.P_values.min()
        self.P_max = self.P_values.max()

        self.rho_interp = interp.interp1d(np.log10(self.xws_values), np.log10(self.rho_values))
        self.ne_interp = interp.interp1d(np.log10(self.xws_values),  np.log10(self.ne_values))
        self.P_interp = interp.interp1d(np.log10(self.xws_values),   np.log10(self.P_values))
        self.mufe_interp = interp.interp1d(np.log10(self.xws_values), self.mufe_values)
        self.rho_P_interp = interp.interp1d(np.log10(self.P_values), np.log10(self.rho_values))
        self.xws_P_interp = interp.interp1d(np.log10(self.P_values), np.log10(self.xws_values))

        
    def xws_P(self, P):
        if P > self.P_max or P < self.P_min:
            return -1
        else:
            return 10**self.xws_P_interp(np.log10(P))

    
    def rho(self, xws):

        '''
        Returns density in kg/m^3
        '''
        if xws<self.xmin or xws>self.xmax:
            return 0
        else:
            return 10**self.rho_interp(np.log10(xws))*gTOkg/cmTOm**3
    
    def ne(self, xws):
        '''
        Returns ne in pm^-3
        '''
        if xws<self.xmin or xws>self.xmax:
            return 0
        else:
            return 10**self.ne_interp(np.log10(xws))

    def np(self, xws):
        '''
        Returns np in pm^-3
        '''
        if xws < self.xmin or xws > self.xmax:
            return 0
        else:
            return  3 / (4 * pi * (xws * lampi * inveVTOm * mTOpm)**3) # Z

    def P(self, xws):
        '''
        Returns P in Pa
        '''
        if xws<self.xmin or xws>self.xmax:
            return 1e-20
        else:
            return 10**self.P_interp(np.log10(xws))

    def mufe(self, xws):
        '''
        Returns muFe in MeV
        '''
        if xws<self.xmin or xws>=self.xmax:
            return 0
        else:
            return self.mufe_interp(np.log10(xws))

    def rho_P(self, P):
        '''
        Returns density in kg/m^3 as a function of pressure
        '''
        if P>self.P_values.iloc[0] or P<self.P_values.iloc[-1]:
            return 1e-20

        else:
            return 10**self.rho_P_interp(np.log10(P))*gTOkg/cmTOm**3

    def P_prime(self, xws):

        # Depreciated

        # self.sep = (self.xws_values.iloc[2] - self.xws_values.iloc[1])
        # self.P_grid = self.P_values#[self.P(xx) for xx in self.xws_values]
        # # self.P_grad = np.gradient(self.P_grid)
        # self.log_P_grad = np.gradient(self.P_grid, self.sep)
        # print(self.log_P_grad)
        # # self.neg_log_P_prime = interp.UnivariateSpline(np.log10(self.xws_values), np.log10(-self.P_grad))
        # self.log_P_prime = interp.UnivariateSpline(np.log10(self.xws_values), self.log_P_grad)

        if (xws-self.delta_x*1.1)<self.xmin or (xws+0.9*self.delta_x)>self.xmax:
            return 0
        else:            
            # return -10**self.neg_log_P_prime(np.log10(xws))
            # return self.log_P_prime(np.log10(xws))
            
            delta_P = (self.P(xws + self.delta_x) - self.P(xws-self.delta_x))
            # if np.isnan(delta_P/self.delta_x):
            #     print('bad at {}'.format(xws))
            return delta_P/self.delta_x/2.
    
    
class TOV:

    def __init__(self, element, temperature):
        self.elem = element
        self.temp = temperature

        self.eos = eos_class(element, temperature)

    def M_P0_rel(self, elem, M):
        
        # Returns the lP0 (log of central pressure) for the given mass

        file = 'results/{}/{:0.3e}/mass_rad/MR.dat'.format(elem, self.temp)
        data = pd.read_csv(file, header=0, sep = '\t')
        data = data.drop(data[data['M'] > 1.4].index)

        max_loc = data['M'].idxmax()
        M_xws0 = interp1d(data['M'][max_loc:], data['log_10_P0'][max_loc:])

        return M_xws0(M)

    def lP0_of_R(self, elem, rad):
        file = 'results/{}/{:0.3e}/mass_rad/MR.dat'.format(elem, self.temp)
        data = pd.read_csv(file, header=0, sep='\t')

        # max_loc = data['M'].idxmax()
        M_xws0 = interp1d(data['R'], data['log_10_P0'])

        return M_xws0(rad)

    def P0_M_rel(self, elem, P0):
        file = 'results/{}/{:0.3e}/mass_rad/MR.dat'.format(elem, self.temp)
        data = pd.read_csv(file, header=0, sep='\t')

        M = interp1d(data['log_10_P0'], data['M'])

        return M(np.log10(P0))

    def TOV_system(self, x, y):
        '''
        Input:
        -------
        x = Radial coordinate
        y = (m, P)

        Outputs:
        ---------
        [dmdr, dPdr]
        '''
        r = x
        m = y[0]
        # xws = y[1]
        P = y[1]
        phi = y[2]

        # P = self.eos.P(xws)
        # dens = self.eos.rho(xws)
        # P_prime = self.eos.P_prime(xws)
        dens = self.eos.rho_P(P)

        dmdr = 4 * pi * r**2 * dens
        dphidr = (m + 4*pi * r**3 * P/cspeed**2)/r/(r - 2* GNewt * m/cspeed**2)
        # dxwsdr = -GNewt *(dens + P/cspeed**2)*dphidr/P_prime
        dPdr =  -GNewt *(dens + P/cspeed**2)*dphidr
        
        # return [dmdr, dxwsdr]
        return [dmdr, dPdr, dphidr]


    def TOV_solver(self, P0, dr):
        
        r0 = dr

        m0 = 4.*pi*self.eos.rho_P(P0)*r0**3/3.
        # P = self.eos.P(xws0)
        P = P0
        phi_0 = 0.0

        r_vals = np.array([r0])
        m_vals = np.array([m0])
        P_vals = np.array([P])
        xws_now = self.eos.xws_P(P)
        rho_vals = np.array([self.eos.rho_P(P)])
        ne_vals = np.array([self.eos.ne(xws_now)])
        np_vals = np.array([self.eos.np(xws_now)])
        mufe_vals = np.array([self.eos.mufe(xws_now)])
        phi_vals = np.array([phi_0])


        # Pp = self.eos.P_prime(xws0)

        r_now = r0
        y0 = [m0, P, phi_0]
        count = 0

        delta_m = 0
        while True:
            
            if (P > self.eos.P_max or P < self.eos.P_min):
                if count>1:
                    print("Out of EoS range, delta_mass = {} after {} steps".format(delta_m, count))
                break

            TOV_sol = integrate.solve_ivp(self.TOV_system, [r_now, r_now+dr], y0, method = 'RK45')

            r_vals   = np.append(r_vals,   TOV_sol.t[-1])
            m_vals   = np.append(m_vals,   TOV_sol.y[0, -1])
            P_vals   = np.append(P_vals,   TOV_sol.y[1, -1])
            phi_vals = np.append(phi_vals, TOV_sol.y[2,-1])

            count = count+1
            P = P_vals[-1]
            r_now = r_now + dr
            xws_now = self.eos.xws_P(P)

            rho_vals  = np.append(rho_vals, self.eos.rho_P(P))
            ne_vals   = np.append(ne_vals, self.eos.ne(xws_now))
            np_vals   = np.append(np_vals, self.eos.np(xws_now))
            mufe_vals = np.append(mufe_vals, self.eos.mufe(xws_now))

            y0 =  [m_vals[-1], P_vals[-1], phi_vals[-1]]
            delta_m = np.fabs(m_vals[count] - m_vals[count-1])/(m_vals[count] + m_vals[count-1])

            # if  delta_m < 1e-11:
            #     print("Reached a stable mass of {:0.5} M_sol after {} steps".format(m_vals[-1]/Msol, count))
            #     break
            # print(P, delta_m)

        # phi_R = 0.5*np.log(1 - (2. * GNewt * m_vals[-1])/ (cspeed**2 * r_vals[-1]))
        # phi0 = phi_vals[-1]
        # phi_vals_new_1 = phi_vals - phi0
        # phi_vals_new = phi_vals_new_1 + phi_R 

        # B_vals = np.exp(2.*phi_vals_new)
        B_vals = B_prof_sover(r_vals/1000, m_vals/Msol, P_vals*10)
                    
        return r_vals, m_vals, rho_vals, P_vals, ne_vals, np_vals, mufe_vals, B_vals


    def Mass_Radius(self, lstart, lstop,  num = 100, make_profs = True):
        '''
        Inputs:
        --------
        lstart: Start solving MR relation with central xWS 10^lstart
        lstop:  Stop solving MR relation with central xWS 10^lstop
        dr: Step size of TOV solver. Default is 10
        num: Number of points of Mass Radius relation. Default 100

        Outputs:
        ---------
        R: Radius in m
        M: Mass of star in M_sol
        '''

        lP0_vals = np.linspace(lstart, lstop, num)

        y1 = 1.7
        y2 = 3.4

        dr = lambda lP0: 10**(-(y2-y1)*(lP0 - lstart)/(lstop - lstart) + y2)
        # dr = 1e4
        print("\nSolving MR for {} WD at T={:0.3e}K".format(self.elem, self.temp))

        MR_vals = []
        Teff_vals = np.array([])
        surf_g_vals = np.array([])
        
        for lP in lP0_vals:
            R_vals, M_vals, rho_vals, P_vals, ne_vals, np_vals, mufe_vals, B_vals  = self.TOV_solver(10**lP, dr(lP))
            print("\tResults: M = {}, R = {}\n".format(M_vals[-2]/Msol, R_vals[-2]))
            MR_vals.append([M_vals[-2], R_vals[-2]])

            if make_profs == True:
                mstar = M_vals[-2]/Msol

                profile_data = pd.DataFrame({'r[m]': R_vals[:-1], 'm[Msun]': np.array(M_vals[:-1])/Msol, 'rho[kg/m^3]': rho_vals[:-1], 'P[Pa]': P_vals[:-1], 'ne[pm^-3]': ne_vals[:-1], 'np[pm^-3]': np_vals[:-1], 'muF_e[MeV]': mufe_vals[:-1], 'B': B_vals[:-1]})
                if not os.path.isdir('results/{}/{:0.3e}/profiles/{:0.3f}'.format(self.elem, self.temp, mstar)):
                    os.system('mkdir -p results/{}/{:0.3e}/profiles/{:0.3f}'.format(self.elem, self.temp, mstar))
                # profile_data.to_csv('profiles/profile_{}_{:0.3e}_{:0.3f}.dat'.format(self.elem, self.temp, mstar), index=False, sep='\t')
                profile_data.to_csv('results/{}/{:0.3e}/profiles/{:0.3f}/profiles.dat'.format(self.elem, self.temp, mstar), index=False, sep='\t')
                # Teff_vals = np.append(Teff_vals, T_eff(ne_vals[:-1], 2.0 * np_vals[:-1], self.eos.elem.Mnuc, self.temp, R_vals[:-1], M_vals[-2]))
                surf_g_vals = np.append(surf_g_vals, surf_gravity(R_vals[-2], M_vals[-2]))


        masses = np.array([m[0]/Msol for m in MR_vals])
        radii = np.array([r[1] for r in MR_vals])


        MR_data = pd.DataFrame({'R': radii, 'M': masses, 'log_10_P0': lP0_vals, 'g_surf_(m/s^2)': surf_g_vals})
        if not os.path.isdir('results/{}/{:0.3e}/mass_rad'.format(self.elem, self.temp)):
            os.system('mkdir -p results/{}/{:0.3e}/mass_rad'.format(self.elem, self.temp))
        # MR_data.to_csv('Mass_Radius/MR_{}_{:0.3e}.dat'.format(self.elem, self.temp), index = False, sep = '\t')
        MR_data.to_csv('results/{}/{:0.3e}/mass_rad/MR.dat'.format(self.elem, self.temp), index = False, sep = '\t')
        

        return np.array(masses), np.array(radii), lP0_vals

class mixed_WD:

    def __init__(self, elemInner, elemOutter, temp):
        self.elemInner = elemInner
        self.elemOutter = elemOutter
        self.temp = temp

        self.eosInner = eos_class(elemInner, temp)
        self.eosOutter = eos_class(elemOutter, temp)

    def TOV_system(self, x, y, eos):
        '''
        Input:
        -------
        x = Radial coordinate
        y = (m, P)
        eos = the EOS class

        Outputs:
        ---------
        [dmdr, dPdr, dphidr]
        '''
        r = x
        m = y[0]
        # xws = y[1]
        P = y[1]
        phi = y[2]

        # P = self.eos.P(xws)
        # dens = self.eos.rho(xws)
        # P_prime = self.eos.P_prime(xws)
        dens = eos.rho_P(P)

        dmdr = 4 * pi * r**2 * dens
        dphidr = (m + 4*pi * r**3 * P/cspeed**2) /r/(r - 2 * GNewt * m/cspeed**2)
        # dxwsdr = -GNewt *(dens + P/cspeed**2)*dphidr/P_prime
        dPdr = -GNewt * (dens + P/cspeed**2)*dphidr

        # return [dmdr, dxwsdr]
        return [dmdr, dPdr, dphidr]

    def M_P0_rel(self, M):

        file = 'results/{}{}/{:0.3e}/mass_rad/MR.dat'.format(self.elemInner, self.elemOutter, self.temp)
        data = pd.read_csv(file, header=0, sep = '\t')

        data = data.drop(data[data['M'] > 1.4].index)

        max_loc = data['M'].idxmax()
        min_loc = data['M'].idxmin()

        M_xws0 = interp1d(data['M'][min_loc:max_loc], data['log_10_P0'][min_loc:max_loc])
        return M_xws0(M)

    def P0_M_rel(self, elem, P0):
        file = 'results/{}/{:0.3e}/mass_rad/MR.dat'.format(elem, self.temp)
        data = pd.read_csv(file, header=0, sep='\t')

        M = interp1d(data['log_10_P0'], data['M'])

        return M(np.log10(P0))

    def R_from_M(self, M):
        # print(self.temp)
        file = 'results/{}{}/{:0.3e}/mass_rad/MR.dat'.format(self.elemInner, self.elemOutter, self.temp)
        data = pd.read_csv(file, header=0, sep='\t')

        max_loc = data['M'].idxmax()
        M_xws0 = interp1d(data['M'][:max_loc], data['R'][:max_loc])

        return M_xws0(M)

    def TOV_central(self, P0, dr):

        M_star = self.P0_M_rel(self.elemInner, P0)

        r_vals = np.logspace(np.log10(dr/1e7), np.log10(dr), 1000)

        r0 = r_vals[0]
        m0 = 4.*pi*self.eosInner.rho_P(P0)*r0**3/3.
        # P = self.eosInner.P(xws0)
        P = P0
        phi_0 = 0.0

        # r_vals    = np.array([r0])
        m_vals    = np.array([m0])
        P_vals    = np.array([P])
        xws_now   = np.array(self.eosInner.xws_P(P))
        rho_vals  = np.array([self.eosInner.rho_P(P)])
        ne_vals   = np.array([self.eosInner.ne(xws_now)])
        np_vals   = np.array([self.eosInner.np(xws_now)])
        mufe_vals = np.array([self.eosInner.mufe(xws_now)])
        phi_vals  = np.array([phi_0])

        y0 = [m0, P, phi_0]
        count = 0

        print("Solving central region of a {} M_sun WD, with P_0 = {} and T = {:0.3e} K:".format(M_star, P0, self.temp))

        for ii, r_now in enumerate(r_vals[:-1]):
            TOV_sol = integrate.solve_ivp(self.TOV_system, [r_now, r_vals[ii + 1]], y0, method = 'RK45', args = (self.eosInner,))
            # print('here')

            m_vals   = np.append(m_vals, TOV_sol.y[0, -1])
            P_vals   = np.append(P_vals, TOV_sol.y[1, -1])
            phi_vals = np.append(phi_vals, TOV_sol.y[2, -1])

            # print(m_vals, P_vals)

            count = count+1

            P = P_vals[-1]
            xws_now = self.eosInner.xws_P(P)

            rho_vals  = np.append(rho_vals, self.eosInner.rho_P(P))
            ne_vals   = np.append(ne_vals,  self.eosInner.ne(xws_now))
            np_vals   = np.append(np_vals, self.eosInner.np(xws_now))
            mufe_vals = np.append(mufe_vals, self.eosInner.mufe(xws_now))

            y0 =  [m_vals[-1], P_vals[-1], phi_vals[-1]]
            # delta_m = np.fabs(m_vals[count] - m_vals[count-1])/(m_vals[count] + m_vals[count-1])

        B_vals = B_prof_sover(np.array(r_vals)/1000.0, np.array(m_vals)/Msol, np.array(P_vals)*10.0)
        
        return r_vals, m_vals, rho_vals, P_vals, ne_vals, np_vals, mufe_vals, B_vals



        
    def TOV_mixed(self, P0, dr):

        # lxws0 = np.log10(xws0)
        # xws0 = self.eosInner.xws_P(P0)

        M_star = self.P0_M_rel(self.elemInner, P0)

        # dr = 10**((3.2-2)*(lxws0 - 2)/(3 - 2) + 2)

        r0 = dr
        m0 = 4.*pi*self.eosInner.rho_P(P0)*r0**3/3.
        # P = self.eosInner.P(xws0)
        P = P0
        phi_0 = 0.0

        r_vals    = np.array([r0])
        m_vals    = np.array([m0])
        P_vals    = np.array([P])
        xws_now   = np.array(self.eosInner.xws_P(P))
        rho_vals  = np.array([self.eosInner.rho_P(P)])
        ne_vals   = np.array([self.eosInner.ne(xws_now)])
        np_vals   = np.array([self.eosInner.np(xws_now)])
        mufe_vals = np.array([self.eosInner.mufe(xws_now)])
        phi_vals  = np.array([phi_0])

        r_now = r0
        y0 = [m0, P, phi_0]
        count = 0

        print("Solving for a {} M_sun WD, with P_0 = {} and T = {:0.3e} K:".format(M_star, P0, self.temp))

        while (m_vals[-1] <= M_star*Msol/2):
            TOV_sol = integrate.solve_ivp(self.TOV_system, [r_now, r_now+dr], y0, method = 'RK45', args = (self.eosInner,))

            r_vals   = np.append(r_vals, TOV_sol.t[-1])
            m_vals   = np.append(m_vals, TOV_sol.y[0, -1])
            P_vals   = np.append(P_vals, TOV_sol.y[1, -1])
            phi_vals = np.append(phi_vals, TOV_sol.y[2, -1])
            count = count+1

            r_now = r_now + dr
            P = P_vals[-1]
            xws_now = self.eosInner.xws_P(P)

            rho_vals  = np.append(rho_vals, self.eosInner.rho_P(P))
            ne_vals   = np.append(ne_vals,  self.eosInner.ne(xws_now))
            np_vals   = np.append(np_vals, self.eosInner.np(xws_now))
            mufe_vals = np.append(mufe_vals, self.eosInner.mufe(xws_now))

            y0 =  [m_vals[-1], P_vals[-1], phi_vals[-1]]
            delta_m = np.fabs(m_vals[count] - m_vals[count-1])/(m_vals[count] + m_vals[count-1])


        m_Inner = m_vals[-1]
        print('\tSwitching to {} EOS at M = {:0.4f} M_sun, R = {:0.4f} m, P = {:0.4e} Pa'.format(self.elemOutter, m_vals[-1]/Msol, r_vals[-1], P_vals[-1]))
        # P_half = P_vals[-1]
        r_transition = r_vals[-1]

        while True:
            # if (P > self.eosOutter.P_max or P < self.eosOutter.P_min):
            if xws_now == -1:
                # if (m_vals[-1] >= M_star*Msol):
                m_Outter = m_vals[-1]-m_Inner
                m_ratio = m_Inner/m_Outter
                if count>1:
                    print("\tOut of EoS range, delta_mass = {} after {} steps".format(delta_m, count))
                    print("\tM_reached/M_wanted = {:0.4f}, m_{}/m_{} = {:0.4f}".format(m_vals[-1]/Msol/M_star,  self.elemInner, self.elemOutter, m_ratio))
                break

            TOV_sol = integrate.solve_ivp(self.TOV_system, [r_now, r_now+dr], y0, method = 'RK45', args = (self.eosOutter,))

            count = count+1
            P = P_vals[-1]
            r_now = r_now + dr
            xws_now = self.eosOutter.xws_P(P)
            # print(self.eosOutter.P_min, self.eosOutter.P_max, P/self.eosOutter.P_min, xws_now)

            r_vals   = np.append(r_vals, TOV_sol.t[-1])
            m_vals   = np.append(m_vals, TOV_sol.y[0, -1])
            P_vals   = np.append(P_vals, TOV_sol.y[1, -1])
            phi_vals = np.append(phi_vals, TOV_sol.y[2, -1])

            rho_vals  = np.append(rho_vals, self.eosOutter.rho_P(P))
            ne_vals   = np.append(ne_vals,  self.eosOutter.ne(xws_now))
            np_vals   = np.append(np_vals, self.eosOutter.np(xws_now))
            mufe_vals = np.append(mufe_vals, self.eosOutter.mufe(xws_now))

            y0 =  [m_vals[-1], P_vals[-1], phi_vals[-1]]
            delta_m = np.fabs(m_vals[count] - m_vals[count-1])/(m_vals[count] + m_vals[count-1])

        

        # if m_vals[-1]>= M_star*Msol:
        #     print("\t\tDidin't reach EOS end for {}, with dm = {}".format(M_star, delta_m))

        B_vals = B_prof_sover(np.array(r_vals)/1000.0, np.array(m_vals)/Msol, np.array(P_vals)*10.0)
        
        return r_vals, m_vals, rho_vals, P_vals, ne_vals, np_vals, mufe_vals, B_vals, r_transition

    def Mass_Radius(self, lstart, lstop,  num=25, make_profs = True):
        '''
        Inputs:
        --------
        lstart: Start solving MR relation with central xWS 10^lstart
        lstop:  Stop solving MR relation with central xWS 10^lstop
        # dr: Step size of TOV solver. Default is 10
        num: Number of points of Mass Radius relation. Default 100

        Outputs:
        ---------
        R: Radius in m
        M: Mass of star in M_sol
        '''

        lP_vals = np.linspace(lstart, lstop, num)

        y1 = 1.7
        y2 = 3.4

        def dr(lP0): return 10**(-(y2-y1)*(lP0 - lstart)/(lstop - lstart) + y2)
        
        MR_vals = []

        print("\nSolving MR for {}{} WD at T={:0.3e}K".format(self.elemInner, self.elemOutter, self.temp))
        
        for lP in lP_vals:
        
            R_cent, M_cent, rho_cent, P_cent, ne_cent, np_cent, mufe_cent, B_cent = self.TOV_central(10**lP, dr(lP))
            R_vals, M_vals, rho_vals, P_vals, ne_vals, np_vals, mufe_vals, B_vals, r_tran  = self.TOV_mixed(10**lP, dr(lP))
        
            print("\tResults: M = {}, R = {}\n".format(M_vals[-1]/Msol, R_vals[-1]))
        
            MR_vals.append([M_vals[-1], R_vals[-1]])
            # print(MR_vals)

            if make_profs == True:
                mstar = M_vals[-1]/Msol

                profile_data = pd.DataFrame({'r[m]': R_vals[:-1], 'm[Msun]': np.array(M_vals[:-1])/Msol, 'rho[g/cm^3]': np.array(rho_vals)[:-1]*kgTOg/(mTOcm**3), 'P[Pa]': P_vals[:-1], 'ne[pm^-3]': ne_vals[:-1], 'np[pm^-3]': np_vals[:-1], 'muF_e[MeV]': mufe_vals[:-1], 'B': B_vals[-1]})
                if not os.path.isdir('results/{}{}/{:0.3e}/profiles/{:0.3f}'.format(self.elemInner, self.elemOutter, self.temp, mstar)):
                    os.system('mkdir -p results/{}{}/{:0.3e}/profiles/{:0.3f}'.format(self.elemInner, self.elemOutter, self.temp, mstar))
                # profile_data.to_csv('profiles/profile_{}{}_{:0.3e}_{:0.3f}.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), index=False, sep='\t')
                profile_data.to_csv('results/{}{}/{:0.3e}/profiles/{:0.3f}/profiles.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), index=False, sep='\t')
                np.savetxt('results/{}{}/{:0.3e}/profiles/{:0.3f}/r_transition.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), [r_tran])

                trans_data = np.transpose((R_cent[:-1], np.array(M_cent)[:-1]/Msol, np.array(rho_cent)[:-1]*kgTOg/(mTOcm**3), P_cent[:-1], ne_cent[:-1], np_cent[:-1], mufe_cent[:-1], B_cent[:-1]))
                # print(trans_data)
                np.savetxt('results/{}{}/{:0.3e}/profiles/{:0.3f}/profile_central.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), trans_data, delimiter='\t', header='r[m]\tm[Msun]\trho[g/cm^3]\tP[Pa]\tne[pm^-3]\tnp[pm^-3]\tmuF_e[MeV]\tB')



        masses = [m[0]/Msol for m in MR_vals]
        radii = [r[1] for r in MR_vals]



        MR_data = pd.DataFrame({'R': radii, 'M': masses, 'log_10_P0': lP_vals})
        if not os.path.isdir('results/{}{}/{:0.3e}/mass_rad'.format(self.elemInner, self.elemOutter, self.temp)):
            os.system('mkdir -p results/{}{}/{:0.3e}/mass_rad'.format(self.elemInner, self.elemOutter, self.temp))
        # MR_data.to_csv('Mass_Radius/MR_{}{}_{:0.3e}.dat'.format(self.elemInner, self.elemOutter, self.temp), index=False, sep='\t')
        MR_data.to_csv('results/{}{}/{:0.3e}/mass_rad/MR.dat'.format(self.elemInner, self.elemOutter, self.temp), index=False, sep='\t')

        

        return np.array(masses), np.array(radii), lP_vals


    def Mass_Radius_Mscan(self, lstart, lstop,  num=25, make_profs = True):
        '''
        Inputs:
        --------
        lstart: Start solving MR relation with central xWS 10^lstart
        lstop:  Stop solving MR relation with central xWS 10^lstop
        # dr: Step size of TOV solver. Default is 10
        num: Number of points of Mass Radius relation. Default 100

        Outputs:
        ---------
        R: Radius in m
        M: Mass of star in M_sol
        '''

        lM_vals = np.linspace(lstart, lstop, num)

        y1 = 1.7
        y2 = 3.4

        lstartP = 21.5
        lstopP = 30

        def dr(lP0): return 10**(-(y2-y1)*(lP0 - lstartP)/(lstopP - lstartP) + y2)
        
        MR_vals = []

        print("\nSolving MR for {}{} WD at T={:0.3e}K".format(self.elemInner, self.elemOutter, self.temp))
        
        for mm in lM_vals:

            try:
                lP = self.M_P0_rel(mm)
            except:
                continue
        
            R_cent, M_cent, rho_cent, P_cent, ne_cent, np_cent, mufe_cent, B_cent = self.TOV_central(10**lP, dr(lP))
            R_vals, M_vals, rho_vals, P_vals, ne_vals, np_vals, mufe_vals, B_vals, r_tran  = self.TOV_mixed(10**lP, dr(lP))
        
            print("\tResults: M = {}, R = {}\n".format(M_vals[-1]/Msol, R_vals[-1]))
        
            MR_vals.append([M_vals[-1], R_vals[-1]])
            # print(MR_vals)

            if make_profs == True:
                mstar = M_vals[-1]/Msol

                profile_data = pd.DataFrame({'r[m]': R_vals[:-1], 'm[Msun]': np.array(M_vals[:-1])/Msol, 'rho[g/cm^3]': np.array(rho_vals)[:-1]*kgTOg/(mTOcm**3), 'P[Pa]': P_vals[:-1], 'ne[pm^-3]': ne_vals[:-1], 'np[pm^-3]': np_vals[:-1], 'muF_e[MeV]': mufe_vals[:-1], 'B': B_vals[-1]})
                if not os.path.isdir('results/Berg/{}{}/{:0.3e}/profiles/{:0.3f}'.format(self.elemInner, self.elemOutter, self.temp, mstar)):
                    os.system('mkdir -p results/Berg/{}{}/{:0.3e}/profiles/{:0.3f}'.format(self.elemInner, self.elemOutter, self.temp, mstar))
                # profile_data.to_csv('profiles/profile_{}{}_{:0.3e}_{:0.3f}.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), index=False, sep='\t')
                profile_data.to_csv('results/Berg/{}{}/{:0.3e}/profiles/{:0.3f}/profiles.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), index=False, sep='\t')
                np.savetxt('results/Berg/{}{}/{:0.3e}/profiles/{:0.3f}/r_transition.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), [r_tran])

                trans_data = np.transpose((R_cent[:-1], np.array(M_cent)[:-1]/Msol, np.array(rho_cent)[:-1]*kgTOg/(mTOcm**3), P_cent[:-1], ne_cent[:-1], np_cent[:-1], mufe_cent[:-1], B_cent[:-1]))
                # print(trans_data)
                np.savetxt('results/Berg/{}{}/{:0.3e}/profiles/{:0.3f}/profile_central.dat'.format(self.elemInner, self.elemOutter, self.temp, mstar), trans_data, delimiter='\t', header='r[m]\tm[Msun]\trho[g/cm^3]\tP[Pa]\tne[pm^-3]\tnp[pm^-3]\tmuF_e[MeV]\tB')



        masses = [m[0]/Msol for m in MR_vals]
        radii = [r[1] for r in MR_vals]



        # MR_data = pd.DataFrame({'R': radii, 'M': masses, 'log_10_P0': lP_vals})
        # if not os.path.isdir('results/{}{}/{:0.3e}/mass_rad'.format(self.elemInner, self.elemOutter, self.temp)):
        #     os.system('mkdir -p results/{}{}/{:0.3e}/mass_rad'.format(self.elemInner, self.elemOutter, self.temp))
        # # MR_data.to_csv('Mass_Radius/MR_{}{}_{:0.3e}.dat'.format(self.elemInner, self.elemOutter, self.temp), index=False, sep='\t')
        # MR_data.to_csv('results/{}{}/{:0.3e}/mass_rad/MR.dat'.format(self.elemInner, self.elemOutter, self.temp), index=False, sep='\t')

        

        return np.array(masses), np.array(radii)

def MR_plotter(plot_by = 'elem', elem = 'C', temp = 1e5):
    MR_files = glob('results/*/**/mass_rad/MR.dat')
    # print(MR_files)
    # file_filter(elem, temp, MR_files)

    fig, ax = plt.subplots()

    if plot_by == 'T':
        for element in ['He', 'C', 'O']:
            data = pd.read_csv(file_filter(element, temp, MR_files), header = 0, sep = '\t')
            ax.plot(data['R']/Rsol/0.01, data['M'], label = '{}'.format(element))      
            legend_title= '{:0.3e}'.format(temp)
        

    elif plot_by == 'elem':
        for i, T in enumerate(np.append(np.zeros(1), np.logspace(4, 8, 5))):
            data = pd.read_csv(file_filter(elem, float(T), MR_files), header = 0, sep = '\t')
            ax.plot(data['R']/Rsol/0.01, data['M'], label = '{:0.3e} K'.format(T))
            legend_title = elem
            
    ax.set_xlabel(r'$R_\star/(0.01\,R_\odot)$')
    ax.set_ylabel(r'$M_\star/M_\odot$')
    # plt.xlim(1, 6)
    # plt.ylim(0.05, 1)
    # plt.yscale('log')
    ax.legend(title=legend_title, ncol=2, prop={'size': 6})
    # plt.show()
    return fig, ax

if __name__ == "__main__":

        # for elem in ['C', 'O']:
    T0 = 0
    T1 = np.logspace(4, 8, 21)
    # T2 = np.logspace(7, 8, 10)
    # T_tot = np.append(T1, T2[1:-1])
    # T_tot = np.append(T0, T1)
    T_tot = T1

    # T_tot = np.logspace(4, 8, 5)

    lp0_1 = 20
    lp0_2 = 30
    num = 25

    mass1 = 0.2
    mass2 = 1.3
    mass_inc = 0.05
    mass_num = int( (mass2 - mass1)/mass_inc) + 1

    for T in T_tot:
    #     for elem in ['C']:
    # # # # elem = 'O'
    # # # # T = 10000
    # #         # print(T)
    #         TOV_single = TOV(elem, T)
    #         TOV_single.Mass_Radius(lp0_1, lp0_2, num)

        TOV_mixed = mixed_WD('O', 'C', T)
        TOV_mixed.Mass_Radius_Mscan(mass1, mass2, mass_num)
        # TOV_mixed.Mass_Radius(lp0_1,lp0_2, num)

        # TOV_mixed = mixed_WD('C', 'He', T)
        # TOV_mixed.Mass_Radius(lp0_1, lp0_2, num)

    ########################################
    # Single WD
    ########################################

    # RS = 1241.3811457394036e3
    # Mstar = 0.622
    # # elem = 'O'
    # # temps = np.append(np.array([0.0]), np.logspace(4, 7, 10))

    # temps = [0.0] #10**np.array([0])
    # lp0_1 = 21.5
    # lp0_2 = 30
    # num = 25

    
    
    # for temp in temps:

    #     # if temp == 0.0 or np.log10(temp) in [0, 4, 5, 6, 7, 8]:
    #     #     print(temp)
    #     #     continue

    #     y1 = 1.7
    #     y2 = 3.4

    #     def dr(lP0): return 10**(-(y2-y1)*(lP0 - 21.5)/(30.0 - 21.5) + y2)

    #     system = mixed_WD('C', 'O', temp)
    #     lP0 = system.M_P0_rel(Mstar)
    #     dr_set = dr(lP0)

    #     r_vals, m_vals, rho_vals, P_vals, ne_vals, np_vals, mufe_vals, B_vals, r_trans = system.TOV_mixed(10**lP0, dr_set)

    #     trans_data = np.transpose([r_vals[:-1], np.array(m_vals)[:-1]/Msol, np.array(rho_vals)[:-1]*kgTOg/(mTOcm**3), P_vals[:-1], ne_vals[:-1], np_vals[:-1], mufe_vals[:-1], B_vals[:-1]])
    #     np.savetxt(f'special_WDs/{Mstar}_Msun/FMT_OC_{temp:0.1e}K.dat', trans_data, header='r[m]\tM[Msun]\trho[g/cm^3]\tP[N/m^2]\tn_e[pm^-3]\tn_p[pm^-3]\tmuF_e[MeV]\tB', delimiter='\t')
    #     np.savetxt(f'special_WDs/{Mstar}_Msun/r_transition_{temp:0.1e}K.dat', np.array([r_trans]))

        # TOV_single = TOV('O', temp)
        # TOV_single.Mass_Radius(lp0_1, lp0_2, num)

        # TOV_mixed = mixed_WD('O', 'C', temp)
        # TOV_mixed.Mass_Radius(lp0_1, lp0_2, num)

