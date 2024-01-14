import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.special import erf
from bsm_formula import bsm_formula, normdist

def early_exercise_boundary_gbm(vol, r, q, T, K, phi=1,
        n_steps=100,xtol=1e-3):
    """
    Computes an approximation of the early exercise boundary of an american
    option given n_steps time steps, under the Geometric Brownian Motion
    Assumption.
    """

    # Time step
    dt = T/n_steps

    # Initializing the boundary
    # We only need an n_steps sized vector but it matches the indices and we
    # drop the first entry at exit
    boundary = np.zeros(n_steps+1)

    # Setting the boundary at maturity
    if q == 0:
        # Without dividends
        boundary[-1] = K
    elif phi == 1:
        # Put with dividends
        boundary[-1] = min(K, r*K/q)
    else:
        # Call with dividends
        boundary[-1] = max(K, r*K/q)

    # Helper functions for the log-normal distribution
    d1 = lambda S, t, B, u: (np.log(S/B) + (r - q +
        0.5*(vol**2))*(u-t))/(vol*np.sqrt(u-t))

    d2 = lambda S, t, B, u: d1(S,t,B,u) - vol*np.sqrt(u-t)

    # Recursive iteration
    for j in range(n_steps-1,0,-1):

        tau = T - j*dt
        
        # Defining the implicit equation
        # European option term
        def european(B):
            p1 = K*np.exp(-r*tau)*normdist(-phi*d2(B,j*dt,K,T))
            p2 = B*np.exp(-q*tau)*normdist(-phi*d1(B,j*dt,K,T))

            return phi*(p1 - p2)

        # Early exercise premium term
        def eep(B):
            # Simpson's rule for first step
            #if j == n_steps-1:
            #    pass
            #    midpoints = [B+k/8*(boundary[-1]-B) for k in range(9)]
            #    premium = r*K-q*B
            #    t1 = r*K*dt*np.exp(-r*dt)
            #    n1 = normdist(-phi*d2(B,j*dt,boundary[-1],T))
            #    t2 = q*B*np.exp(-q*dt)
            #    n2 = normdist(-phi*d1(B,j*dt,boundary[-1],T))
            #    increment = t1*n1+t2*n2
            #    premium += increment
            #    for l in range(7):
            #        h = dt/8
            #        t1 = r*K*np.exp(-r*h*(l+1))
            #        n1 = normdist(-phi*d2(B,j*dt,midpoints[l],j*dt+h*(l+1)))
            #        t2 = q*B*np.exp(-r*h*(l+1))
            #        n1 = normdist(-phi*d2(B,j*dt,midpoints[l],j*dt+h*(l+1)))
            #        increment = 2*(t1*n1 - t2*n2)
            #        if l%2==0:
            #            increment = 2*increment
            #        premium += increment
            #    premium = premium*h/3
            #    return premium
            premium = phi*(r*K*dt-q*B*dt)/2
            for l in range(j+1,n_steps+1):
                t1 = r*K*dt*np.exp(-r*dt*(l-j))
                n1 = normdist(-phi*d2(B,j*dt,boundary[l],l*dt))

                t2 = q*B*dt*np.exp(-q*dt*(l-j))
                n2 = normdist(-phi*d1(B,j*dt,boundary[l],l*dt))

                increment = phi*(t1*n1 - t2*n2)
                if l==n_steps:
                    increment = increment/2
                premium += increment
            return premium
        
        # Implicit equation for Bj
        implicit = lambda B: european(B) + eep(B) + phi*(B - K)

        # The optimization is problematic in the first few steps
        # The solution I found involves limiting the number of iterations to a
        # low number early on, which still guarantees convergence but avoids the
        # appearence of spikes close to the boundary
        boundary[j] = fsolve(implicit, boundary[-1],maxfev=(8+(n_steps-j)))
    return boundary[1:]


def recursive_pricing_gbm(S0, vol, r, q, T, K, phi, n_steps):
    """
    Implements the full recursive method based on the integral representation
    approach for pricing American options under the Geometric Brownian Motion
    assumption.
    """
    # Helper functions for the log-normal distribution
    d1 = lambda S, t, B, u: (np.log(S/B) + (r - q +
        0.5*(vol**2))*(u-t))/(vol*np.sqrt(u-t))

    d2 = lambda S, t, B, u: d1(S,t,B,u) - vol*np.sqrt(u-t)

    # Obtaining the early exercise boundary
    boundary = early_exercise_boundary_gbm(vol,r,q,T,K,phi,n_steps)

    # European option term
    european = bsm_formula(S0,vol,r,q,T,K,phi)

    # Early exercise premium term
    dt = T/n_steps
    eep = 0

    for j in range(1,n_steps+1):
        t1 = r*K*np.exp(-r*j*dt)*normdist(-phi*d2(S0,0,boundary[j-1],j*dt))
        t2 = q*S0*np.exp(-r*j*dt)*normdist(-phi*d1(S0,0,boundary[j-1],j*dt))

        eep += phi*(t1 - t2)*dt

    option_value = european + eep
    return option_value, european, eep, boundary


def test(phi,n_steps):
    """
    Test function for showcasing the script.
    Computes the early exercise boundary of a tipical put, plots it and computes
    the option value.
    """
    start_time = time.time()
    option_price, eu, eep, boundary = \
        recursive_pricing_gbm(100,0.25,0.05,0.06,0.5,100,phi,n_steps)
    print("Running time: %.2f seconds" % (time.time() - start_time))
    print("American option value: %.4f" % option_price)
    print("European part of value: %.4f" % eu)
    print("Early exercise premium: %.4f" % eep)
    t = np.linspace(0.5/n_steps,0.5,n_steps)
    plt.plot(t,boundary)
    plt.show()
     








