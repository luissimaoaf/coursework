import numpy as np
from scipy.special import erf
import time
import matplotlib.pyplot as plt
from bsm_formula import bsm_formula

# Model parameters

# Spot
S = 100
# Volatility
vol = 0.25
# Interest rate
r = 0.03    
# Dividend yield
q = 0

# Contract parameters

# Time to maturity
T = 0.5    
# Strike
K = 100    


def binomial_cox(S, vol, r, q, T, K, phi, N):
    """
        binomial_cox:
    """

    # Setting the parameters
    # Time step
    dt = T/N
    # Square root of time step
    sqrtdt = np.sqrt(dt)               
    # Up move
    u = np.exp(vol*sqrtdt)
    # Down move
    d = np.exp(-vol*sqrtdt)
    # Up probability
    qu = (np.exp((r-q)*dt) - d)/(u - d)
    # Down probability
    qd = 1 - qu                         

    # Initialization
    asset_grid = np.zeros([N+1, N+1])
    option_grid = np.zeros([N+1, N+1])
    
    # Generating the asset grid
    asset_grid[0,0] = S
    for i in range(N+1):
        for j in range(i+1):
            asset_grid[i,j] = S*(u**j)*(d**(i-j))
    
    # Setting the payoff at maturity
    for j in range(N+1):
        option_grid[-1,j] =  max(phi*(K - asset_grid[-1,j]), 0)

    # Backwards induction
    for i in range(N,0,-1):
        for j in range(i):
            option_grid[i-1,j] = np.exp(-r*dt) * (qu*option_grid[i,j+1] +
                qd*option_grid[i,j])
    
    return option_grid[0,0]


## Initialization ##
# Script parameters
grids = [5, 10, 20, 50, 100, 1000, 2000, 5000, 10000]
n_seq = len(grids)

call_cox = np.zeros(n_seq)
put_cox = np.zeros(n_seq)

## Main loop ##
start_time = time.time()

for k in range(n_seq):

    # Setting the number of time steps
    N = grids[k]

    call = binomial_cox(S, vol, r, q, T, K, -1, N)
    put = binomial_cox(S, vol, r, q, T, K, 1, N)

    call_cox[k] = call
    put_cox[k] = put

call_bsm = bsm_formula(S, vol, r, q, T, K, -1)
put_bsm = bsm_formula(S, vol, r, q, T, K, 1)


cerrorcox = np.log(np.abs(call_cox - call_bsm))[1:]
perrorcox = np.log(np.abs(put_cox - put_bsm))[1:]

print("--- %.4f seconds ---" % (time.time() - start_time))

print("\n--- Binomial Call Prices ---")
print(call_cox)
print("\n--- BSM Call Price ---")
print(call_bsm)

print("\n----------------------")

print("\n--- Binomial Put Prices ---")
print(put_cox)
print("\n--- BSM Put Price ---")
print(put_bsm)

xpoints = grids[1:]

plt.figure()
plt.plot(xpoints, cerrorcox, '-o', label="Binomial Call Price")
plt.plot(xpoints, perrorcox, '-o', label="Binomial Put Price")

plt.title("Convergence to BSM prices")
plt.xlabel("Number of iterations")
plt.ylabel("Log-absolute error")
plt.legend()

plt.savefig('binomialcox-convergence.png')















