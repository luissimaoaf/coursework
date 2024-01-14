import numpy as np
from scipy.special import erf, gamma, gammaincc
from scipy.stats import ncx2


def normdist(x):
    """
    normdist
    Computes the commulative density function of the normal distribution
    """

    N = (1.0 + erf(x/np.sqrt(2.0)))/2.0

    return N


def delta_bsm(s,q,r,vol,k,tau,phi):
    """
    delta_bsm
    s: spot
    q: dividend yield
    r: risk-free interest rate
    vol: stock volatility
    k: strike
    tau: time to maturity
    phi: 1 if put, -1 if call
    """

    d1 =  (np.log(s/k) + (r - q + 0.5*vol**2)*tau)/(vol*np.sqrt(tau))
    delta = phi * (-np.exp(-q*tau) * normdist(-phi*d1))

    return delta


def delta_cev(s,q,r,sigma,beta,k,tau,phi):
    """
    greeks.delta_cev
    s: spot
    q: dividend yield
    r: risk-free interest rate
    sigma: volatility factor
    beta: elasticity parameter
    k: strike
    tau: time to maturity
    phi: 1 if put, -1 if call
    """
    
    if beta == 2:
        # GBM assumption
        delta = delta_bsm(s,q,r,sigma,k,tau,phi)

        return delta
    
    # Computing parameters
    kappa = 2*(r-q)/(sigma**2 * (2-beta) * (np.exp((r-q)*(2-beta)*tau)-1))
    x = kappa * (s**(2-beta)) * np.exp((r-q)*(2-beta)*tau)
    y = kappa * k**(2-beta)

    if beta < 2:
        # Martingale case
        if phi == -1:
            # Call delta
            f1 = np.exp(-q*tau) * (1 - ncx2.cdf(2*y, 2 + 2/(2-beta), 2*x))
            f2 = s * np.exp(-q*tau) * ncx2.pdf(2*y, 4 + 2/(2-beta), 2*x)
            f3 = k * np.exp(-r*tau) * ncx2.pdf(2*x, 2/(2-beta), 2*y)

            delta = f1 + 2*x*(2 - beta)/s * (f2 - f3)

        else:
            # Put delta
            f1 = np.exp(-q*tau) * ncx2.cdf(2*y, 2 + 2/(2-beta), 2*x)
            f2 = s * np.exp(-q*tau) * ncx2.pdf(2*y, 4 + 2/(2-beta), 2*x)
            f3 = k * np.exp(-r*tau) * ncx2.pdf(2*x, 2/(2-beta), 2*y)

            delta = -f1 + 2*x*(2 - beta)/s * (f2 - f3)

    else:
        # Bubble case
        if phi == -1:
            # Call delta
            f1 = np.exp(-q*tau) * (1 - ncx2.cdf(2*x, 2/(beta-2), 2*y))
            f2 = s * np.exp(-q*tau) * ncx2.pdf(2*x, 2/(beta-2), 2*y)
            f3 = k * np.exp(-r*tau) * ncx2.pdf(2*y, 4 + 2/(beta-2), 2*x)
            
            # Bubble term
            v = 1/(beta-2)
            bubble = np.exp(-q*tau) * gammaincc(v,x) + \
                    np.exp(-q*tau-x) * (x**v) / gamma(v+1)
            delta = f1 - 2*x*(2-beta)/s * (f2 - f3) - bubble

        else:
            # Put delta
            f1 = np.exp(-q*tau) * ncx2.cdf(2*x, 2/(beta-2), 2*y)
            f2 = s * np.exp(-q*tau) * ncx2.pdf(2*x, 2/(beta-2), 2*y)
            f3 = k * np.exp(-r*tau) * ncx2.pdf(2*y, 4 + 2/(beta-2), 2*x)
            
            delta = -f1 - 2*x*(2-beta)/s * (f2 - f3)

    return delta






