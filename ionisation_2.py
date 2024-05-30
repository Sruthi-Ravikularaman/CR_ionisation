from ionization_tools import integrate, velocity


m_p = 938.272  # in MeV
m_e = 0.511  # in MeV



def diff_H2_ion_rate_p(E, f_p, sig_p, sig_ec, phi_p, L_e):
    '''
        Differential proton ionization rate.
        Parameters
        ----------
        E MeV
        f_p function, MeV-1 cm-3
        sig_p function, cm2
        sig_ec function, cm2
        phi_p function

        Returns
        -------
        dzeta MeV-1 s-1

        '''

    v, _, _ = velocity(E, m_p)
    dzeta_p = v * f_p(E) * sig_p(E) * (1 + phi_p(E, L_e))
    dzeta_ec = v * f_p(E) * sig_ec(E)

    dzeta = dzeta_p + dzeta_ec

    return dzeta


def diff_H2_ion_rate_e(E, f_e, sig_e, phi_e, L_e):
    '''
        Differential electron ionization rate.
        Parameters
        ----------
        E MeV
        f_e function, MeV-1 cm-3
        sig_e function, cm2
        phi_e function

        Returns
        -------
        dzeta MeV-1 s-1

        '''

    v, _, _ = velocity(E, m_e)
    f = f_e(E)
    sig = sig_e(E)
    phi = phi_e(E, L_e)
    dzeta = v * f * sig * (1 + phi)

    return dzeta


def H2_ion_rate_p(E_min, E_max, f_p, sig_p, sig_ec, phi_p, L_e):
    '''
    Proton ionization rate.
    Parameters
    ----------
    E_min MeV
    E_max MeV
    f_p function, MeV-1 cm-3
    sig_p function, cm2
    sig_ec function, cm2
    phi_p function

    Returns
    -------
    zeta s-1

    '''
    def integrand_p(E):
        v, _, _ = velocity(E, m_p)
        return v * f_p(E) * sig_p(E) * (1 + phi_p(E, L_e))

    def integrand_ec(E):
        v, _, _ = velocity(E, m_p)
        return v * f_p(E) * sig_ec(E)
    
    def integrand(E):
        return integrand_p(E) + integrand_ec(E)

    zeta = integrate(integrand, E_min, E_max)

    return zeta


def H2_ion_rate_e(E_min, E_max, f_e, sig_e, phi_e, L_e):
    '''
    Electron ionization rate.
    Parameters
    ----------
    E_min MeV
    E_max MeV
    f_e function, MeV-1 cm-3
    sig_e function, cm2
    phi_e function

    Returns
    -------
    zeta s-1

    '''

    def integrand_e(E):
        v, _, _ = velocity(E, m_e)
        return v * f_e(E) * sig_e(E) * (1 + phi_e(E, L_e))

    zeta = integrate(integrand_e, E_min, E_max)

    return zeta