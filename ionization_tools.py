'''
This file has some functions that are recurrently used forward.

'''
import numpy as np
import math


m_p = 938.272  # in MeV
m_e = 0.511  # in MeV
c = 3e10  # in cm/s, velocity of light


def integrate_1(f, x_min, x_max, N_pts = 500):
    '''
    Integrates by changing variables x -> log(x).

    Parameters
    ----------
    f : function
        Function to be integrated.
    x_min : float
        Lower limit of integration.
    x_max : float
        Upper limit of integration.
    N_pts : int, optional
        Number of points to take in the logspaced interval of integration.
        The default is 500.

    Returns
    -------
    int_value : float
        Integral value.

    '''
    if x_min==x_max:
        int_value=0
    else:
        if x_min == 0:
            z_min = np.log10(1e-50)
        else:
            z_min = np.log10(x_min)
        z_max = np.log10(x_max)
        int_range = np.logspace(z_min, z_max, N_pts)
        int_range = np.log(int_range)
        int_value = 0
        for i in range(len(int_range)-1):
            middle = (int_range[i]+int_range[i+1])/2
            y_middle = np.exp(middle)*f(np.exp(middle))
            int_value += y_middle*(int_range[i+1]-int_range[i])
    return int_value


def integrate_2(f, x_min, x_max, N_pts = 100):
    g = np.vectorize(f)
    if x_min==x_max:
        int_value=0
    else:
        if x_min == 0:
            z_min = np.log10(1e-50)
        else:
            z_min = np.log10(x_min)
        z_max = np.log10(x_max)
        int_range = np.log(np.logspace(z_min, z_max, N_pts))
        mid_points = (int_range[:-1]+int_range[1:])/2
        y_mid_points = np.exp(mid_points)*g(np.exp(mid_points))
        int_value = np.sum(y_mid_points*np.diff(int_range))
    return int_value


def integrate_trapz(f, x_min, x_max, N_pts, iflog):

    g = np.vectorize(f)
    if iflog:
        x_list = np.logspace(np.log10(x_min), np.log10(x_max), N_pts)
    else:
        x_list = np.linspace(x_min, x_max, N_pts)
    y_list = g(x_list)

    return np.trapz(y_list, x_list)


def decades(xmin, xmax):
    x1, x2  = np.log10(xmin), np.log10(xmax)
    d1, d2 = math.floor(x1), math.ceil(x2)
    return d2 - d1


def integrate(f, xmin, xmax, nptd=10):
    g = np.vectorize(f)
    if xmin == xmax:
        int_value =  0.0   
    else:
        if xmin == 0.0:
            ln_min= np.log(0.1) 
            d_min = np.log10(0.1)
        else:
            ln_min= np.log(xmin)
            d_min = np.log10(xmin)
        ln_max= np.log(xmax)
        d_max = np.log10(xmax)

        d_list = np.array([d_min] + list(range(math.floor(d_min)+1, math.ceil(d_max))) + [d_max])

        xd_list = []
        for d1, d2 in zip(d_list[:-1], d_list[1:]):
            xd_list.append(np.logspace(d1, d2, nptd))
        xd_list = sorted(list(set(np.array(xd_list).flatten())))

        ln_x = np.log(xd_list)
        x = np.exp(ln_x)
        y = g(x)
        int_value = np.trapz(y*x, ln_x)
        
    return int_value


def step_integrate(f, x_min, x_max, step):
    '''
    Integrates by changing variables x -> log(x).

    Parameters
    ----------
    f : function
        Function to be integrated.
    x_min : float
        Lower limit of integration.
    x_max : float
        Upper limit of integration.
    step : float, optional
        Step to take in the linspaced interval of integration.
        The default is 500.

    Returns
    -------
    int_value : float
        Integral value.

    '''
    if x_min==x_max:
        int_value=0
    else:
        if x_min == 0:
            z_min = np.log10(1e-50)
        else:
            z_min = np.log10(x_min)
        z_max = np.log10(x_max)
        N_pts = math.ceil((z_max-z_min)/step)+1
        int_range = np.linspace(z_min, z_max, N_pts)
        int_range = np.power(10, int_range)
        int_range = np.log(int_range)
        int_value = 0
        for i in range(len(int_range)-1):
            middle = (int_range[i]+int_range[i+1])/2
            y_middle = np.exp(middle)*f(np.exp(middle))
            int_value += y_middle*(int_range[i+1]-int_range[i])
    return int_value


def velocity(E_kin, rest_mass):
    '''
    Gives the velocity, it's ratio with c and Lorentz factor.

    Parameters
    ----------
    E_kin : float
        Kinetic energy of particle in MeV.
    rest_mass : float
        Rest mass energy of particle in MeV.

    Returns
    -------
    v : float
        Particle velocity in cm s^-1.
    beta : float
        v/c.
    gamma : float
        Lorentz factor.

    '''
    gamma = 1+(E_kin/rest_mass)
    beta = math.sqrt(1-(1/(gamma**2)))
    v = c*beta
    return v, beta, gamma


def vel_p(p, m):
    E = np.sqrt(p ** 2 + m ** 2) - m
    gamma = 1 + (E / m)
    beta = math.sqrt(1 - (1 / (gamma ** 2)))
    v = c * beta
    return v, beta, gamma

