import time
import numpy as np
from scipy.special import erf
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


def binomial_bs(S, vol, r, q, T, K, phi, N):
    """
    binomial_bs:
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
    
    # No need to compute the payoff at maturity
    # Computing BSM prices at time step before maturity
    for j in range(N):
        option_grid[-2,j] =  bsm_formula(asset_grid[-2,j], vol, r, q, dt, K,
                phi) 

    # Backwards induction
    for i in range(N-1,0,-1):
        for j in range(i):
            option_grid[i-1,j] = np.exp(-r*dt) * (qu*option_grid[i,j+1] +
                qd*option_grid[i,j])
    
    return option_grid[0,0]


## Initialization ##
# Script parameters
# Doubling length for Richardson interpolation
grids = [5*2**n for n in range(12)]
n_seq = len(grids)

call_bbs = np.zeros(n_seq)
put_bbs = np.zeros(n_seq)

call_bsm = bsm_formula(S, vol, r, q, T, K, -1)
put_bsm = bsm_formula(S, vol, r, q, T, K, 1)

start_time = time.time()

## Main loop ##

for k in range(n_seq):

    # Setting the number of time steps
    N = grids[k]

    call = binomial_bs(S, vol, r, q, T, K, -1, N)
    put = binomial_bs(S, vol, r, q, T, K, 1, N)

    call_bbs[k] = call
    put_bbs[k] = put

call_bbsr = [2*call_bbs[k+1] - call_bbs[k] for k in range(n_seq-1)]
put_bbsr = [2*put_bbs[k+1] - put_bbs[k] for k in range(n_seq-1)]


cerrorbbs = np.log(np.abs(call_bbs - call_bsm))[1:]
cerrorbbsr = np.log(np.abs(call_bbsr - call_bsm))

perrorbbs = np.log(np.abs(put_bbs - put_bsm))[1:]
perrorbbsr = np.log(np.abs(put_bbsr - put_bsm))

print("--- %.4f seconds ---" % (time.time() - start_time))

print("\n--- BBS Call Prices ---")
print(call_bbs)
print("\n--- BBSR Call Prices ---")
print(call_bbsr)
print("\n--- BSM Call Price ---")
print(call_bsm)

print("\n----------------------")

print("\n--- BBS Put Prices ---")
print(put_bbs)
print("\n--- BBSR Put Prices ---")
print(put_bbsr)
print("\n--- BSM Put Price ---")
print(put_bsm)

xpoints = grids[1:]

fig1 = plt.figure()

plt.plot(xpoints, cerrorbbs, '-o', label="Binomial BS")
plt.plot(xpoints, cerrorbbsr, '-o', label="BBS with Richardson Extrapolation")

plt.title("Convergence to BSM call price")
plt.xlabel("Number of iterations")
plt.ylabel("Log-absolute error")
plt.legend()

plt.savefig('bbs-convergence-call.png')
plt.close(fig1)

fig2 = plt.figure()

plt.plot(xpoints, perrorbbs, '-o', label="Binomial BS")
plt.plot(xpoints, perrorbbsr, '-o', label="BBS with Richardson Extrapolation")

plt.title("Convergence to BSM put price")
plt.xlabel("Number of iterations")
plt.ylabel("Log-absolute error")
plt.legend()

plt.savefig('bbs-convergence-put.png')







