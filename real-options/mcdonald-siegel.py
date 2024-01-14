"""
mcdonald-siegel.py
Replicates tables 1 and 2 on McDonald and Siegel (1986), by computing the
optimal investment level and the current investment value for the given ranges
of parameters.
Saves the complete results as CSV files and prints the formatted tables matching
the ones on the paper.
"""

import time
import numpy as np


def vol2(sigmav=0.2, sigmaf=0.2, rho=0.0):
    """
    Auxiliary function for total variance
    Base values as predetermined inputs
    """
    vol = sigmav**2 + sigmaf**2 - 2*rho*sigmav*sigmaf
    return vol


def epsilon(sigma2, lamb=0.0, deltav=0.1, deltaf=0.1):
    """
    Auxiliary function for computing epsilon
    Base values as predetermined inputs
    """
    t1 = 0.5 - (deltaf-deltav)/sigma2
    eps = np.sqrt(t1**2 + 2*(deltaf + lamb)/sigma2) + t1

    return eps


def cstar(eps):
    """
    Auxiliary function for optimal investment level
    """
    return eps/(eps-1)


def investment_value(v0, f0, eps):
    """
    Auxiliary function for investment value
    """
    c = cstar(eps)
    x = (c-1)*f0*(v0/(c*f0))**eps

    return x


# Range of values for computation
delvvals = [0.05,0.1,0.25]
delfvals = [0.01,0.05,0.1,0.25]
rhovals = [-0.5,0,0.5]
sigvals = np.sqrt([0.01,0.02,0.04,0.1,0.2,0.3])
lambvals = [0.0,0.05,0.1,0.25]

# Saving lengths for looping
ndelv = len(delvvals)
ndelf = len(delfvals)
nrho = len(rhovals)
nsig = len(sigvals)
nlamb = len(lambvals)

# Table sizes
n_cols = nrho*ndelv
n_rows = nsig + nlamb + ndelf

# Tables for replicating the results
table1 = np.zeros((n_rows,n_cols))
table2 = np.zeros((n_rows,n_cols))

# Main loop
for k in range(ndelv):
    deltav = delvvals[k]

    for i in range(nrho):
        rho = rhovals[i]
        
        for j in range(nsig):
            sigmav = sigvals[j]
            sigmaf = sigvals[j]

            sigma2 = vol2(sigmav=sigmav,sigmaf=sigmaf,rho=rho)
            eps = epsilon(sigma2,deltav=deltav)
            c = cstar(eps)            
            val = investment_value(1,1,eps)
            
            table1[j,k*ndelv+i] = val
            table2[j,k*ndelv+i] = c

        for j in range(nlamb):
            lamb = lambvals[j]

            sigma2 = vol2(rho=rho)
            eps = epsilon(sigma2,lamb=lamb,deltav=deltav)
            c = cstar(eps)
            val = investment_value(1,1,eps)

            table1[nsig+j,k*ndelv+i] = val
            table2[nsig+j,k*ndelv+i] = c

        for j in range(ndelf):
            deltaf = delfvals[j]

            sigma2 = vol2(rho=rho)
            eps = epsilon(sigma2,deltav=deltav,deltaf=deltaf)
            c = cstar(eps)
            val = investment_value(1,1,eps)

            table1[nsig+nlamb+j,k*ndelv+i] = val
            table2[nsig+nlamb+j,k*ndelv+i] = c

# Saving the results as CSV files
np.savetxt('table1.csv',table1,delimiter=',')
np.savetxt('table2.csv',table2,delimiter=',')

# Rounding the values to 2 decimal places
table1 = np.round(table1,2)
table2 = np.round(table2,2)

# Printing
print(table1)
print(table2)











