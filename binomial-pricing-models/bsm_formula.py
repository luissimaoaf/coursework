"""
bsm_formula.py
Auxiliary functions - cumulative density function for the normal distribution
and the Black-Scholes formula for European option pricing.
"""

from scipy.special import erf
import numpy as np


def normdist(x):
    """
        normdist
    """

    N = (1.0 + erf(x/np.sqrt(2.0)))/2.0

    return N


def bsm_formula(S, vol, r, q, T, K, phi):
    """
        bsm_formula
    """

    d1 = (np.log(S/K) + (r - q + 0.5*vol**2)*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    n1 = normdist(-phi*d1)
    n2 = normdist(-phi*d2)

    option_price = phi*(np.exp(-r*T)*K*n2 - np.exp(-q*T)*S*n1)

    return option_price


