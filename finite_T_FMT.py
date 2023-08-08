# User defined data
from constants import *

# Required Modules
import numpy as np
from scipy import integrate
import pandas as pd
from scipy import interpolate as interp
import scipy.special as spec
# from matplotlib import pyplot as plt
import os

#=====================================================================

#=====================================================================
# Misc. Functions
#=====================================================================


def archive_cells(dir, remove=True):

    prevdir = os.getcwd()
    # dir = 'results/{}/{:0.3e}/cells/'.format(elem, T)
    os.system('rm -rf {}cell_data.tgz'.format(dir))

    os.chdir('{}cells/'.format(dir))

    if remove == True:
        comm = 'tar -zcvf cell_data.tgz FMT_cell*.dat --remove-files'
    else:
        comm = 'tar -zcvf cell_data.tgz FMT_cell*.dat'

    # print(comm)
    os.system(comm)
    os.chdir(prevdir)

def clean_cell_data(dir):
    os.system('rm -r {}FMT_cell*'.format(dir))

#=====================================================================
# Element fucntions
#=====================================================================


class elem_data:
    '''
    Class to access element data
    Znum = Atomic number, Z
    Mnuc = Attomic mass, A
    mufrac = N_p/Z
    Delta, Rc and xc are defined in arxiv:1312.2434
    '''
    
    def __init__(self, elem, T):

        if elem == "He":
            self.Mnuc = 4.003
            self.mufrac = 2.
            self.Znum = 2.

        elif elem == "C":
            self.Mnuc = 12.01
            self.mufrac = 2.
            self.Znum = 6.

        elif elem == "O":
            self.Mnuc = 16
            self.mufrac = 2
            self.Znum = 8

        self.Delta = r0*np.cbrt(self.Mnuc/self.Znum)/lampi
        self.Rc = self.Delta*lampi*np.cbrt(self.Znum)
        self.xc = self.Rc/lampi


#=====================================================================
# Fermi-Dirac Integrals
#=====================================================================

# def eta:
#     eta_prefac = 1/(lampi * mE*beta)
#     return eta_prefac*(chi/x)

# def finite_FD_integrand(t, x, chi, beta, k):
#     return t**k * np.sqrt( 1 + 0.5*beta * t)/(1 + np.exp(t - eta(x, chi, beta)))

# # def chi_deriv_finite_FD_integrand(t, x, chi, k):
# #     return (t**k * np.sqrt(1 + 0.5*t*beta) * eta_prefac)/(2*x + 2*x*np.cosh(t - eta_prefac*chi/x))

# def finite_FD(x, chi, beta, k):
#     integrand = lambda t: t**k * np.sqrt( 1 + 0.5*beta * t)/(1 + np.exp(t - eta(x, chi, beta)))
#     res = integrate.quad(integrand, 0, max(0, 10+eta(x, chi, beta)))[0]
#     return res


def gen_FD_f(t, beta, k):
    '''
    Generalised Fermi-Dirac numerator
    '''
    return t**k * np.sqrt(1 + beta * t * 0.5)


def gen_FD_int(t, eta, beta, k):
    '''
    Generalised Fermi-Dirac integrand
    '''
    return gen_FD_f(t, beta, k)/(np.exp(t-eta) + 1)


def delta_f(t, x, chi, beta, k):
    '''
    Function used to evaluate Fermi-Dirac integrals following procedure in Fukushima: "Computation of a general integral of Fermi–Dirac distribution by McDougall–Stoner method"
    '''
    eta = chi/(mE*beta*lampi*x)

    # denom = (2*( (-t + eta)**k * np.sqrt(1 + (beta*(-t + eta))/2) + (t + eta)**k * np.sqrt(1 + (beta*(t + eta))/2)))

    if k == 1/2:
        numer = 2*(t + beta*eta*t)
        denom = np.sqrt(1 + (beta*(eta - t))/2)*np.sqrt(eta - t) + np.sqrt(eta + t)*np.sqrt(1 + (beta*(eta + t))/2)
    elif k == 3/2:
        numer =  2*t*(3*eta**2 + 2*beta*eta**3 + t**2 + 2*beta*eta*t**2)
        denom = np.sqrt(1 + (beta*(eta - t))/2)*(eta - t)**(3/2) + (eta + t)**(3/2)*np.sqrt(1 + (beta*(eta + t))/2)
    elif k ==5/2:
        numer = 2*t*(5*eta**4 + 3*beta*eta**5 + 10*eta**2*t**2 + 10*beta*eta**3*t**2 + t**4 + 3*beta*eta*t**4)
        denom = np.sqrt(1 + (beta*(eta - t))/2)*(eta - t)**(5/2) + (eta + t)**(5/2)*np.sqrt(1 + (beta*(eta + t))/2)
    else:
        return (-((2 + beta*eta)*((-t + eta)**(2*k) - (t + eta)**(2*k))) + t*beta*((-t + eta)**(2*k) + (t + eta)**(2*k)))/(2*((-t + eta)**k*np.sqrt(1 + (beta*(-t + eta))/2) + (t + eta)**k*np.sqrt(1 + (beta*(t + eta))/2)))

    return numer/denom

def delta_f_reg(t, x, chi, beta, k):
    '''
    Function used to evaluate Fermi-Dirac integrals following procedure in Fukushima: "Computation of a general integral of Fermi–Dirac distribution by McDougall–Stoner method"
    '''
    eta = chi/(mE*beta*lampi*x)
    return gen_FD_f(eta+t, beta, k) - gen_FD_f(eta-t, beta, k)


def FD_A(x, chi, beta, k):
    '''
    Function used to evaluate Fermi-Dirac integrals following procedure in Fukushima: "Computation of a general integral of Fermi–Dirac distribution by McDougall–Stoner method"
    '''
    eta = chi/(mE*beta*lampi*x)
    return (eta)**(1+k) * spec.hyp2f1(-1/2, 1+k, 2+k, -beta*eta/2)/(1+k)


def FD_B(x, chi, beta, k):
    '''
    Function used to evaluate Fermi-Dirac integrals following procedure in Fukushima: "Computation of a general integral of Fermi–Dirac distribution by McDougall–Stoner method"
    '''
    int_cut = 60
    eta = chi/(mE*beta*lampi*x)
    if eta == 0:
        return 0
    else:
        def integrand(t): return delta_f_reg(t, x, chi, beta, k)/(np.exp(t)+1)
        res = integrate.quad(integrand, 0, np.min([eta, int_cut]), limit=200, epsabs=1e-4, epsrel=1e-4)[0]
        return res


def FD_C(x, chi, beta, k):
    '''
    Function used to evaluate Fermi-Dirac integrals following procedure in Fukushima: "Computation of a general integral of Fermi–Dirac distribution by McDougall–Stoner method"
    '''
    int_cut = 60
    eta = chi/(mE*beta*lampi*x)
    if eta > int_cut:
        return 0
    else:
        def integrand(t): return gen_FD_f(t+eta, beta, k)/(np.exp(t) + 1)
        res = integrate.quad(integrand, eta, int_cut)[0]
        return res


def finite_FD(x, chi, beta, k):
    '''
    Finite temperature Fermi-Dirac integration
    '''
    int_cut = 60
    eta = chi/(mE*beta*lampi*x)

    if eta < 0:
        # print("in gen integral for x = {:0.3e}, chi = {:0.3e}, eta = {:0.3e}".format(x, chi, eta))
        def intgrand(t): return gen_FD_int(t, eta, beta, k)
        res = integrate.quad(intgrand, 0, int_cut)[0]
        return res

    elif chi == 0:
        return FD_C(x, chi, beta, k)

    else:# eta > 1e-4:
        return FD_A(x, chi, beta, k)+FD_B(x, chi, beta, k)+FD_C(x, chi, beta, k)
    # else:
    #     return FD_C(x, chi, beta, k)

# def dFdchi(x, chi, beta, k):
#     eta_prefac = 1/(lampi * mE*beta)
#     eta = eta_prefac*(chi/x)
#     integrand = lambda t: (t**k * np.sqrt(1 + 0.5*t*beta) * eta_prefac)/(2*x + 2*x*np.cosh(t - eta))
#     res = integrate.quad(integrand, max(0, eta - 10) , max(0, 10 + eta))[0]
#     return res


def dFdchi(x, chi, beta, k):
    '''
    Derivative w.r.t chi of Fermi-Dirac integral used in Newton's method to solve the ODE
    '''

    eta_pref = 1/(lampi * mE*beta)
    eta = eta_pref*chi/x

    term_min = max(eta-10, 0)
    term_max = max(eta+10, 0)

    
    
    def alpha(t):
        return (np.exp(t-eta)*gen_FD_f(t, beta, k))/(np.exp(t-eta) + 1)**2

    # def delta_alpha(t):
    #     return ((-((2 + beta*eta)*((-t + eta)**(2*k) - np.exp(2*t)*(t + eta)**(2*k))) + t*beta*((-t + eta)**(2*k) + np.exp(2*t)*(t + eta)**(2*k)))*eta_pref)/(np.sqrt(2)*(1 + np.exp(t))*x*(-((-t + eta)**k*np.sqrt(2 - t*beta + beta*eta)) + np.exp(t)*(t + eta)**k*np.sqrt(2 + t*beta + beta*eta)))

    if chi == 0:
        def integrand(t): return alpha(t+eta)/(np.exp(t)+1)
        res = integrate.quad(integrand, 0, 10)[0]

    # elif eta > 0:
    #     def int1(t): return alpha(t)
    #     def int2(t): return delta_alpha(t)/(np.exp(t)+1)
    #     def int3(t): return alpha(t+eta)/(np.exp(t)+1)

    #     I1 = integrate.quad(int1, 0, eta, limit=50, epsrel=1e-4, epsabs=1e-4)[0]
    #     I2 = integrate.quad(int2, 0, np.min([eta, 20]), limit=50, epsrel=1e-4, epsabs=1e-4)[0]

    #     if eta >= 20:
    #         I3 = 0
    #     else:
    #         I3 = integrate.quad(int3, eta, 20, limit=50, epsrel=1e-4, epsabs=1e-4)[0]

    #     return I1+I2+I3

    else:
        if term_min == term_max:
            res = 0.
        else:
            res = integrate.quad(alpha, term_min,term_max)[0]

    return eta_pref*res/x

#=====================================================================
# ODE functions
#=====================================================================

def dwdx(x, chi, Delta, xc, beta):
    '''
    Part of ODE system
    '''

    return -4*pi*alpha*x*(3*np.heaviside(xc - x, 0)/(4*pi*Delta**3) - np.sqrt(2)* (mE/mpion)**3 *beta**(3/2) *(finite_FD(x, chi, beta, 1/2) + beta*finite_FD(x, chi, beta, 3/2) )/pi**2 )

def dgdx(x, chi, beta):
    '''
    Part of ODE system
    '''
    return 4*np.sqrt(2)*mE**3 * x *alpha * beta**(3/2)*( dFdchi(x, chi, beta, 1/2) + beta * dFdchi(x, chi, beta, 3/2) )/(mpion**3 *pi)

#=====================================================================
# EoS system
#=====================================================================

def n_p(x, xc, Znum):
    '''
    Proton number density in WS cell
    '''
    if x > xc:
        return 0
    else:
        return 3*Znum/4./pi/(xc*lampi)**3

def n_e(x, chi, beta):
    '''
    Electron number density in WS cell
    '''
    prefac = 8 * pi * np.sqrt(2.) * mE**3 * np.sqrt(beta**3) / (2. * pi)**3
    return prefac * (finite_FD(x, chi, beta, 1/2) + beta * finite_FD(x, chi, beta, 3/2))


def Edens(x, chi, beta):
    '''
    Electron energy density in WS cell
    '''
    return mE * n_e(x, chi, beta) + np.sqrt(2) * mE**4 * beta**2*np.sqrt(beta) * (finite_FD(x, chi, beta, 3/2) + beta*finite_FD(x, chi, beta, 5/2))/pi**2

def read_FMT_out(filename='FMT_out.dat', head=0):
    '''
    Function used to read output of ODE solver to get x_WS and chi profiles of WS cell
    '''
    data = pd.read_csv(filename, header=head, sep='\t')
    x_data = data['x']
    chi_data = data['chi']

    return x_data, chi_data

class cell_FMT:
    '''
    Class used to calculate cell properties for EoS
    '''

    def __init__(self, M_N, Znum, xC, Beta, filename='FMT_out.dat'):

        self.Mnuc = M_N
        self.Z = Znum
        self.xc = xC
        self.beta = Beta
        self.x_data, self.chi_data = read_FMT_out(filename)
        self.xws_out = self.x_data.iloc[-1]
        self.chi_out = self.chi_data.iloc[-1]
        self.zmin = self.x_data.iloc[0]/self.xws_out

        self.vws = 4*pi*(self.xws_out * lampi)**3 / 3
        self.P_nuc = mE * Beta/self.vws
        self.P_e = 2 * np.sqrt(2) * mE**4 * Beta**2*np.sqrt(Beta)*( finite_FD(self.xws_out, self.chi_out, Beta, 3/2) + 0.5*Beta*finite_FD(self.xws_out, self.chi_out, Beta, 5/2) )/3/pi**2
        self.P_tot = self.P_nuc + self.P_e

        self.chi_interp = interp.interp1d(self.x_data, self.chi_data)

    def muf_e(self, x):
        '''
        Electron chemical potential (minus electron rest mass)
        '''
        return self.chi_interp(x)/lampi/x

    def V_coul(self, x):
        '''
        Coulomb potential
        '''
        return (self.muf_e(x) - self.muf_e(self.xws_out))/echarge

    def E_kin(self):
        '''
        Electron kinetic energy
        '''
        # chi = self.chi_out
        xws = self.xws_out

        integrand = lambda z: z**2 * (Edens(xws*z, self.chi_interp(xws*z), self.beta) - mE * n_e(xws*z, self.chi_interp(xws*z), self.beta) )
        
        if self.xc/xws < 1:
         
            res1 = integrate.quadrature(integrand, self.zmin, self.xc/xws, tol=1e-4, rtol=1e-5, maxiter=200, vec_func=False)[0]

            res2 = integrate.quadrature(integrand, self.xc/xws, 1, tol=1e-4, rtol=1e-5, maxiter=200, vec_func=False)[0]

        else:
            res1 = integrate.quadrature(integrand, self.zmin, 1, tol=1e-4, rtol=1e-5, maxiter=200, vec_func=False)[0]
            res2 = 0

        res = res1 + res2

        return 4*pi*(lampi*xws)**3 * res

    def E_coul(self):
        '''
        Coulomb energy of WS cell
        '''
        xws = self.xws_out

        integrand = lambda z: (n_p(xws*z, self.xc, self.Z) - n_e(xws*z, self.chi_interp(xws*z), self.beta))*self.V_coul(xws*z)

        if self.xc < xws:
            res = integrate.quadrature(integrand, self.xc/xws+1e-6/xws, 1, tol=1e-4, rtol=1e-5, maxiter=200, vec_func=False)[0]
        else:
            res = 0

        return 2*pi*(lampi*xws)**3*echarge*res

    def E_nuc(self):
        '''
        Nuclear energy
        '''
        return self.Mnuc*mP + 1.5* mE * self.beta

    def E_tot(self):
        '''
        Total cell energy
        '''
        return self.E_nuc() + self.E_coul() + self.E_kin()

    def density(self):
        '''
        energy density at given x_WS
        '''
        return self.E_tot()/self.vws



#=====================================================================
# ODE system
#=====================================================================

def ode_sys(z, y, params = []):
    '''
    The ODE system to be solved

    Parameters should be passed in as [Delta, xc, beta, x_WS]
    System is scaled so that z = x/x_WS and solved between z = 0 and 1
    '''
    Delta = params[0]
    xc = params[1]
    beta = params[2]
    xws = params[3]
    x = xws * z

    chi, omega, xi, gamma = y

    f = [omega * xws, dwdx(x, chi, Delta, xc, beta) * xws, gamma * xws, dgdx(x, chi, beta) * xi * xws]
    # if np.isnan(f):
    #     print('something is nan in ode sysetm for x_ws = {}, x = {}'.format(xws))
    # f = [omega * x, dwdx(x, chi, [Delta, xc]) * x, gamma * x, dgdx(x, chi) * xi * x]

    return f


#=====================================================================
# ODE solver
#=====================================================================

# def s0_incriment(current_iter, max_iters, divs):
#     '''
#     Function to set how much the unkown initial condition is incrimented each iteration of Newton's       method
#     '''
#     intervals = np.logspace(-0.5, np.log10(divs), divs)/divs
#     # print(intervals)
#     ratio = current_iter/max_iters
#     diff = intervals - ratio

#     new_diff = np.where(diff > 0, diff, np.inf)
#     min_loc = np.where(new_diff == np.amin(new_diff))[0][0]

#     scale = min([2**(min_loc), 16])  # min([2**min_loc, 100])

#     return scale

def s0_incriment(s1, s2):
    '''
    Function to set how much the unkown initial condition is incrimented each iteration of Newton's       method
    '''
    diff = np.abs(s1-s2)
    sum = np.abs(s1+s2)

    rel_diff = diff/sum

    if rel_diff < 1e-50:
        # print("\tNeton's method beggining to oscillate, decreasing step size of s0...")
        return 10.
    else:
         return 1.


def ode_solver(xws, s0, elem, params = [], tol = 1e-4, max_iters = 1000):
    '''
    Routine to solve the ODE at a given x_WS

    Parameters should be passed in as [Delta, xc, beta]

    The function stores the x_WS and chi profiles for each element and temperature so that the system only needs to be solved once, then the EoS can be recalulated from these files.
    The files are archived as a .tgz

    I have set the amount s0 (the unkown initial condition) is increiment to decrease by a factor of 2 after a certain number of iterations
    This is because eventually the steps are too large and it will oscillate between the same 2 values. This can be changed to be finer tuned.

    xws = cell raduis / lambda_pi
    s0 = unknown initial condition
    elem = Element of choice
    tol = level of tolerance between boundary conditions and found solution
    max_iters = maximum number of iterations for Newton's method before giving up
    '''

    Delta = params[0]
    xc = params[1]
    beta = params[2]
    ode_params = [Delta, xc, beta, xws]
    T=beta*mE/kBMeV

    s0_list = np.empty(0)
    a_list = np.empty(0)
    k_check = 0

    z1 = min([1e-4, 10**(np.floor(np.log10((xc/xws)))-3)])
    zc = xc/xws
    z2 = 1

    # z1 = 0
    # z2 = 6

    nInts = 1000
    z_vals1 = np.linspace(z1, min([zc, 1]),int( nInts/10))
    # delta_z = abs(z_vals1[-2] - z_vals1[-1])
    z_vals2 = np.linspace(zc, z2, nInts)
    # print(z_vals)
    log_name = 'results/{}/{:0.3e}/cells/FMT_cell_{:0.3f}.dat'.format(elem, T, xws)

    if not os.path.isdir('results/{}/{:0.3e}/cells/'.format(elem, T)):
        os.system('mkdir -p results/{}/{:0.3e}/cells/'.format(elem, T))


    while k_check < max_iters:

        s0_list = np.append(s0_list, s0)
        y_init = [0, s0, 0, 1]

        nuc_sol = integrate.solve_ivp(ode_sys, [z1, min([zc, 1])], y_init, t_eval=z_vals1, args=(ode_params,), method='RK45')

        y_nuc = [nuc_sol.y[i][-1] for i in range(len(nuc_sol.y))]
        # print("\t solved in nuc")
        # print("\t edge of nuc soln: {}".format(y_nuc))

        if zc < 1:
            # print(zc)
            sol = integrate.solve_ivp(ode_sys, [zc, z2], y_nuc, t_eval = z_vals2, args=(ode_params,), method='Radau')

        else:
            sol = nuc_sol
        # print("\t solved outside nuc")
        # sol = integrate.solve_ivp(ode_sys, [z1, z2], y_init, t_eval = z_vals2, args=(ode_params,), method='Radau')

        rel_diff = np.abs(sol.y[0][-1] - xws*sol.y[1][-1])
        rel_sum = np.abs(sol.y[0][-1] + xws*sol.y[1][-1])

        a = rel_diff/rel_sum
        a_list = np.append(a_list, a)

        
        if a < tol:

            print('Solution found for xWS = {:0.5e}, chi = {:0.5e}, chi\' = {:0.5e}, s0 = {:0.5e}, after {} iterations'.format(xws, sol.y[0][-1], sol.y[1][-1], s0, k_check+1))

            if zc <1:
                x_out = np.append(nuc_sol.t, sol.t[1:])
                chi_out = np.append(nuc_sol.y[0], sol.y[0][1:])
                dchi_out = np.append(nuc_sol.y[1], sol.y[1][1:])

                # print(x_out)
            else:
                if xws>100:
                    print('\n\tSOMETHING IS NOT RIGHT: xws = {}\nNOT OUTPUTTING FULL RANGE'.format(xws))
                x_out = sol.t
                chi_out = sol.y[0]
                dchi_out = sol.y[1]

            out_data = pd.DataFrame({'x': xws*(x_out), 'chi': chi_out, 'dchidx': dchi_out, 's0': s0})
            # out_data = pd.DataFrame({'x': xws*(sol.t), 'chi': sol.y[0], 'dchidx': sol.y[1]})
            # out_data = pd.DataFrame({'x': xws*10**(sol.t), 'chi': sol.y[0], 'dchidx': sol.y[1]})
            out_data.to_csv('FMT_out.dat', index = False, sep = '\t')
            out_data.to_csv(log_name, index=False, sep='\t')


            break

        elif np.isnan(a) == True:
            print('\tA IS NAN')
            s0_list = np.append(s0_list, s0)
            s0 = s0 + 1e-7
        
        elif sol.status != 0:
            print('\tthere is a status error, stat = {}'.format(sol.status))
            # s0_list = np.append(s0_list, s0)

            # delta_s = (sol.y[0][-1] - xws*sol.y[1][-1])/(sol.y[2][-1] - xws*sol.y[3][-1])

            # s0 = s0 + 5*delta_s
            break

        else:
            s0_list = np.append(s0_list, s0)

            delta_s = (sol.y[0][-1] - xws*sol.y[1][-1])/(sol.y[2][-1] - xws*sol.y[3][-1])
            # print('\t\tRecalculating with xws = {}, chi = {}, chi\' = {}, soln = {}, and s0 = {}'.format(xws, sol.y[0][-1], sol.y[1][-1], a, s0))
            # 
            # inc_scale = s0_incriment(s0_list[max([k_check-2, 0])], s0_list[max([k_check, 0])])
            # if k_check < max_iters/4:
            s0 = s0 - delta_s#/inc_scale

            # elif k_check<max_iters/2:
            #     s0 = s0 - delta_s/2

            # elif k_check < 3*max_iters/2:
            #     s0 = s0 - delta_s/4
            # else:
            #     s0 = s0 - delta_s/8
            

        k_check = k_check + 1

        if k_check % 10 ==0:
            print('\t\t Still calculating after {} iterations, ... |soln - BC| = {}'.format(k_check, a))

        if k_check == max_iters:
            print('\tNo solution found at xws = {:0.5e} after {} iterations, |soln - BC| = {}'.format(xws, k_check, a))

            if zc <=1:
                x_out = np.append(nuc_sol.t, sol.t[1:])
                chi_out = np.append(nuc_sol.y[0], sol.y[0][1:])
                dchi_out = np.append(nuc_sol.y[1], sol.y[1][1:])

                # print(x_out)
            else:
                print('\n\tSOMETHING IS NOT RIGHT: xws = {}\nNOT OUTPUTTING FULL RANGE'.format(xws))
                
                x_out = sol.t
                chi_out = sol.y[0]
                dchi_out = sol.y[1]

            out_data = pd.DataFrame({'x': xws*(x_out), 'chi': chi_out, 'dchidx': dchi_out, 's0': s0})
            # out_data = pd.DataFrame({'x': xws*(sol.t), 'chi': sol.y[0], 'dchidx': sol.y[1]})
            # out_data = pd.DataFrame({'x': xws*10**(sol.t), 'chi': sol.y[0], 'dchidx': sol.y[1]})
            out_data.to_csv('FMT_out.dat', index = False, sep = '\t')
            out_data.to_csv(log_name, index=False, sep='\t')
        
    # np.savetxt('logs/s0_vals/s0_list_{}.txt'.format(xws), s0_list, delimiter=',')

    return s0, sol.status

#=====================================================================
# EoS maker
#=====================================================================

def EoS_maker(elem, T, x1 = -3, x2 = 5, num_scale = 20, ode_iters = 1000, tol=1e-4, calc_EoS=True):
    '''
    This is the main function which makes the EoS by solving the Thomas-Fermi equation for a series of x_WS values.
    You can instead choose to not make the EoS here, and just output the cell profiles then solve the EoS later, set calc_EoS to False

    elem = Element
    T = Temperature in K
    x1, x2 = log10 of x_WS bounds. 
    num_scale = number of points per order of magnitude of x_WS. Making this larger should help with convergence of ODE
    ode_iters = max number of iterations ode_solver does before giving up
    tol = tolerance of solution found by ode solver
    '''
    res_dir = 'results/{}/{:0.3e}/'.format(elem, T)

    clean_cell_data(res_dir)

    print('Solving EoS for {} at {:0.3e}K\n'.format(elem, T))

    xws_vals = np.logspace(x1, x2, round(num_scale*(x2 - x1)))

    n_e_vals = np.empty(0)
    muf_e_vals = np.empty(0)
    P_vals = np.empty(0)
    rho_vals = np.empty(0)


    # print(params)

    s0 = 1
    beta = kBMeV*T/mE
    composition_data = elem_data(elem, T)
    Ar = composition_data.Mnuc
    Znum = composition_data.Znum
    Delta = composition_data.Delta
    xc = composition_data.xc

    ode_params = [Delta, xc, beta]

    num_completed = 0
    for xws in xws_vals:
        snext, stat = ode_solver(xws, s0, elem, ode_params, max_iters = ode_iters, tol = tol)
        if stat != 0:
            break
        if calc_EoS == True:
            cell = cell_FMT(Ar, Znum, xc, beta)

            n_e_vals = np.append(n_e_vals, n_e(cell.xws_out, cell.chi_out, beta)/(inveVTOm*mTOpm)**3)
            muf_e_vals = np.append(muf_e_vals, cell.muf_e(cell.xws_out))
            P_vals = np.append(P_vals, cell.P_tot*evTOkg/inveVTOm/inveVTOs**2)
            rho_vals = np.append(rho_vals, cell.density()*evTOkg*kgTOg/inveVTOm**3/mTOcm**3)

            num_completed += 1 

        s0 = snext


    if calc_EoS == True:
        # EoS_data = pd.DataFrame({'xWS': xws_vals,  'n_e[pm^-3]': n_e_vals, 'P[N/m^2]':P_vals, 'mu_F_e[MeV]': muf_e_vals})
        EoS_data = pd.DataFrame({'xWS': xws_vals[:num_completed], 'rho[g/cm^3]': rho_vals, 'n_e[pm^-3]': n_e_vals, 'P[N/m^2]':P_vals, 'mu_F_e[MeV]': muf_e_vals})
        EoS_data.to_csv('EoS_Files/FMT_{:0.3e}K_{}_EoS.dat'.format(T, elem), sep = '\t', index=False, float_format='%.5e')

        if not os.path.isdir('{}EoS/'.format(res_dir)):
            os.system('mkdir -p {}EoS /'.format(res_dir))
        
        EoS_data.to_csv('{}EoS/FMT_finite_T.dat'.format(res_dir), sep = '\t', index=False, float_format='%.5e')
    archive_cells(res_dir)




#=====================================================================
# main code
#=====================================================================
if __name__ == "__main__":
    temps = np.logspace(4, 8, 21)
    # temps = np.array([10**6.5, 10**7.5])
    elems = ['C']

    for element in elems:
        for T in temps:
            # file = 'EoS_Files/FMT_{:0.3e}K_{}_EoS.dat'.format(T, element)
            # if not os.path.exists(file):
            EoS_maker(element, T, x1 = -3, x2 = 5, num_scale=100, ode_iters=50, tol=1e-4, calc_EoS=True)
            # else:
            #     continue
