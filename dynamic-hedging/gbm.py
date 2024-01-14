import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def simulate(s0, mu, sigma, T, n_steps, n_sims):
    """
    gbm.simulate
    Returns a numpy array of n_sims sample paths of Geometric Brownian Motion
    with length n_steps.
    """
    
    sample = np.zeros((n_sims, n_steps))
    sample[:,0] = s0
    dt = T/n_steps

    for i in range(n_steps-1):

        # Vectorized sampling
        z = np.random.normal(0,np.sqrt(dt), n_sims)
        s = sample[:,i] * np.exp((mu - sigma**2/2)*dt + sigma*z)

        sample[:,i+1] = s
    
    return sample


