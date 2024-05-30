import numpy as np
from ionization_tools import velocity, integrate
from math import pi, sqrt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import h5py

m_p = 938.272  # in MeV
m_e = 0.511  # in MeV
c = 3e10 # in cm/s, velocity of light
mass_ratio = m_p/m_e
alpha = 1/137 # fine structure constant
alpha_prime = 0.87 # parameter in proton ionization
a_0 = 5.292e-9 # in cm, Bohr radius
N = 2 # number of orbital electrons for H2
I = 15.426e-6 # in MeV, ionization potential of H2, binding energy of electron in the H2 molecule
R = 13.6e-6 # in MeV, binding energy of electron in the hydrogen atom
U = 25.68e-6 # in MeV, average orbital kinetic energy of electrons in H2 molecule
#U = 39.603e-6 # in MeV, average orbital kinetic energy of electrons


pc = 3e18
kpc = 1e3 * pc
r_cmz = 200 * pc 
z_cmz = 50 * pc

V_cmz = np.pi * r_cmz**2 * (2 * z_cmz)
M_sol = 2e33 # Solar mass, g
m_H = 1.67e-24 # Hydrogen atom mass, g
m_avg = 1.4 * m_H # Average hydrogen mass, g
M_cmz = 6e7 * M_sol # CMZ mass, g

n_cmz = (M_cmz / m_avg) / V_cmz; # Particle density in the CMZ, cm^-3
print("Particle number density in the CMZ is ", n_cmz, " cm-3")
n_disk = 1.0; # Particle density in the disk, cm^-3
n_out = 1e-2; # Particle density outside the disk, cm^-3

#%% CROSS SECTIONS

#Electron capture (Padovani 2009)

def sigma_ec(E):

    # E in MeV
    E_ev = E*(1e6)
    d_0 = -52.793928
    d_1 = 41.219156
    d_2 = -17.304947
    d_3 = 3.292795
    d_4 = -0.238372
    expo = d_0+(d_1*np.log10(E_ev))+(d_2*(np.log10(E_ev)**2))+(d_3*(np.log10(E_ev)**3))+(d_4*(np.log10(E_ev)**4))
    sigma = pow(10, expo)

    return sigma


#Proton impact (Rudd 1992 and Krause 2015)

def diff_sigma_p(W, E):

    A_1 = 0.96
    A_2 = 1.04
    B_1 = 2.6
    B_2 = 5.9
    C_1 = 0.38
    C_2 = 1.15
    D_1 = 0.23
    D_2 = 0.20
    E_1 = 2.2

    T = E/mass_ratio
    w = W/I
    _,beta,_ = velocity(E, m_p)
    v = np.sqrt(m_e/(2*I))*beta
    Q_max = (E+2*m_p)/(1+(m_p+m_e)**2/(2*m_e*E))
    w_c = (Q_max/I)-(2*v)-(R/(4*I))

    L_1 = C_1*(v**D_1)/(1+E_1*(v**(D_1+4)))
    L_2 = C_2*(v**D_2)

    H_1 = A_1*np.log(1+(T/I))/((v**2)+(B_1/(v**2)))
    H_2 = (A_2/(v**2))+(B_2/(v**4))

    F_1 = L_1 + H_1
    F_2 = L_2*H_2/(L_2+H_2)

    S = 4*pi*(a_0**2)*N*(R/I)**2

    dsig_num = (F_1 + F_2*w)*np.power(1+w, -3)
    dsig_den = 1 + np.exp(alpha_prime*(w-w_c)/v)
    dsig = (S/I)*dsig_num/dsig_den

    return dsig


def sigma_p(E):
    def differential(W):
        return diff_sigma_p(W, E)
    Q_max = (E + 2 * m_p) / (1 + (m_p + m_e) ** 2 / (2 * m_e * E))
    return integrate(differential, 0, Q_max)


#Electron impact (Kim 2000, equation (20))

def sigma_e(E):

    tp = E/m_e # E incident electron energy
    bp = I/m_e # I orbital binding energy oh H2 molecule
    up = U/m_e # U average orbital kinetic energy of the target electron

    bt2 = 1-(1/(1+tp)**2)
    bb2 = 1-(1/(1+bp)**2)
    bu2 = 1-(1/(1+up)**2)

    N_i = 1.173  # integral of df/dw (Kim 2000, equation (4))
    t = E / I

    # Parameters from Kim and Rudd 1994
    c = 1.1262
    d = 6.382
    e = -7.8055
    f = 2.1440

    D_t = ((c/3)*(1-(2/(1+t))**3)) + ((d/4)*(1-(2/(1+t))**4))
    D_t = D_t + ((e/5)*(1-(2/(1+t))**5)) + ((f/6)*(1-(2/(1+t))**6))
    D_t = D_t/2

    term_1 = np.log(bt2/(1-bt2))-bt2-np.log(2*bp)
    term_1 = D_t*term_1

    term_2 = 1-(1/t)
    term_2 = term_2 - ((np.log(t)/(t+1))*((1+2*tp)/((1+tp/2)**2)))
    term_2 = term_2 + ((bp**2/((1+tp/2)**2))*((t-1)/2))
    term_2 = (2 - N_i/N)*term_2

    pre = 4*pi*(a_0**2)*(alpha**4)/(bp*(bt2+bu2+bb2))

    return pre*(term_1+term_2)


def diff_sigma_e(W, E):

    tp = E / m_e  # E incident electron energy
    bp = I / m_e  # I orbital binding energy oh H2 molecule
    up = U / m_e  # U average orbital kinetic energy of the target electron

    bt2 = 1 - (1 / (1 + tp) ** 2)
    bb2 = 1 - (1 / (1 + bp) ** 2)
    bu2 = 1 - (1 / (1 + up) ** 2)

    N_i = 1.173
    t = E / I
    w = W/I

    # Parameters from Kim and Rudd 1994
    c = 1.1262
    d = 6.3982
    e = -7.8055
    f = 2.1440

    def df_dw(y):
        return c * (y ** 3) + d * (y ** 4) + e * (y ** 5) + f * (y ** 6)

    term_1 = (1/(w+1)+1/(t-w))*((1+2*tp)/((1+tp/2)**2))*(((N_i/N)-2)/(1+t))

    term_2 = (1/(1+w)**2 + 1/(t-w)**2 + (bp/(1+tp/2))**2)*(2-(N_i/N))

    term_3 = (np.log(bt2/(1-bt2))-bt2-np.log(2*bp))*(1/(N*(1+w)))*df_dw(1/(1+w))

    pre = 4 * pi * (a_0 ** 2) * (alpha ** 4) / (I * bp * (bt2 + bu2 + bb2))

    return pre*(term_1+term_2+term_3)



#%% AVERAGE SECONDARY IONISATIONS PER PRIMARY

def phi_p(E, L_e):

    def integrand(E_s):
        return sigma_e(E_s)*E_s*diff_sigma_p(E_s, E)/L_e(E_s)
    
    v, beta, gamma = velocity(E, m_p)
    Q_max = (2 * m_e * (beta ** 2) * (gamma ** 2) * (mass_ratio ** 2)) / (
                1 + (2 * gamma * mass_ratio) + (mass_ratio ** 2))
    
    if Q_max > I:
        integ = integrate(integrand, I, Q_max)
    else:
        integ = 0


    return integ/sigma_p(E)


def phi_e(E, L_e):

    def integrand(E_s):
        return sigma_e(E_s)*E_s*diff_sigma_e(E_s, E)/L_e(E_s)
    if E > 3*I:
        integ = integrate(integrand, I, (E-I)/2)
    else:
        integ = 0

    return integ/sigma_e(E)