import numpy as np
from scipy.optimize import fsolve
import time
from bsm_formula import bsm_formula, normdist
import matplotlib.pyplot as plt

def analytic(S0, vol, r, q, tau, K, phi):
    """
    analytic
    Prices an American option by using the analytical approximation of
    Barone-Adesi and Whaley for American calls.
    """

    if phi == 1 and r<=0:
        # Put won't be exercised early
        return bsm_formula(S0,vol,r,q,tau,K,1) 

    if phi == -1 and q <= 0:
        # Call won't be exercised early
        return bsm_formula(S0,vol,r,q,tau,K,-1)
        
    if phi == -1 and r == 0:
        # Avoiding division by 0 in call case
        r = 1e-12
    
    # Computation of seed value
    t1 = 1 - 2*(r - q)/(vol**2)
    t2 = 8*r/(vol**2)
    q_inf = (t1 - phi*np.sqrt(t1**2 + t2))*0.5
    S_inf = K/(1 - 1/q_inf)
    h12 = -((r-q)*tau + phi*2*vol*np.sqrt(tau))*K/(S_inf - K)

    if phi == -1:
        # Call seed
        seed = K + (S_inf - K)*(1 - np.exp(h12))
    else:
        # Put seed
        seed = S_inf + (K - S_inf)*np.exp(h12)

    # Auxiliary parameters and functions
    t3 = 8*r/((vol**2)*(1 - np.exp(-r*tau))) 
    q12 = (t1 - phi*np.sqrt(t1**2 + t3))*0.5
   
    european = lambda S: bsm_formula(S,vol,r,q,tau,K,phi)

    t4 = r - q + 0.5*(vol**2) 
    d1 = lambda S: (np.log(S/K) + t4*tau)/(vol*np.sqrt(tau))
    d2 = lambda S: d1(S) - vol*np.sqrt(tau)

    a12 = lambda S: -phi*S*(1 - np.exp(-q*tau)*normdist(-phi*d1(S)))/q12

    # Finding the critical price
    implicit = lambda S: phi*(K-S)-european(S)-a12(S)

    S_crit = fsolve(implicit, x0=seed, xtol=1e-12)
    if phi*S0 <= phi*S_crit:
        opt_price = phi*(K - S0)
    else:
        opt_price = bsm_formula(S0,vol,r,q,tau,K,phi) + a12(S_crit)*(S0/S_crit)**q12

    return opt_price


