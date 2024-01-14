"""
cev.py
For simulating the CEV process using a first order Euler approximation.
"""

import numpy as np
import gbm
import matplotlib.pyplot as plt

def simulate(s0,mu,sigma,beta,T,n_steps,n_sims):
    """
    cev.simulate
    Returns a numpy array with n_sims simulated paths of length n_steps.
    """
    
    sample = np.zeros((n_sims, n_steps))
    sample[:,0] = s0
    dt = T/n_steps
    
    if beta == 2:
        # Use GBM for exact samples
        sample = gbm.simulate(s0,mu,sigma,T,n_steps,n_sims)
    
    else:
        # Euler discretization
        for i in range(n_steps-1):

            # Vectorized sampling
            z = np.random.normal(0,np.sqrt(dt), n_sims)
            s = sample[:,i]*(1+mu*dt) + sigma*(sample[:,i]**(beta/2))*z

            sample[:,i+1] = s
    
    return sample


