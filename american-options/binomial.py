import numpy as np
from scipy.special import erf
import time
import matplotlib.pyplot as plt
from bsm_formula import bsm_formula


def cox(S, vol, r, q, T, K, phi, N):
    """
    binomial_cox

    Prices an American option by the binomial method of Cox et al.
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

            continuation_value = np.exp(-r*dt) * (qu*option_grid[i,j+1] +
                qd*option_grid[i,j])
            early_exercise = phi*(K - asset_grid[i,j])
            
            option_grid[i-1, j] = max(continuation_value, early_exercise) 
    
    return option_grid[0,0]



def bbs(S, vol, r, q, T, K, phi, N):
    """
    binomial_bs

    Prices an American option by the Binomial Black Scholes method.
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
    
    # At the time step before maturity we compute BSM prices
    for j in range(N):
        
        bsm_price = bsm_formula(asset_grid[-2,j],vol,r,q,dt,K,phi)
        early_exercise = phi*(K - asset_grid[-2,j])
        option_grid[-2,j] = max(bsm_price, early_exercise)


    # Backwards induction
    for i in range(N-1,0,-1):
        for j in range(i):

            continuation_value = np.exp(-r*dt) * (qu*option_grid[i,j+1] +
                qd*option_grid[i,j])
            early_exercise = phi*(K - asset_grid[i,j])
            
            option_grid[i-1, j] = max(continuation_value, early_exercise) 
    
    return option_grid[0,0]


def bbsr(S, vol, r, q, T, K, phi, N):
    """
    Prices an American option by the Binomial Black Scholes method with
    Richardson interpolation.
    """

    price_2n = bbs(S,vol,r,q,T,K,phi,N)
    n = int(N/2)
    price_n = bbs(S,vol,r,q,T,K,phi,n)
    
    bsr_price = 2*price_2n - price_n

    return bsr_price


def test():
	"""
	Showcases the methods implemented above.
	"""
	## Initialization ##
	# Script parameters
	grids = [5*(2**j) for j in range(8)]
	n_seq = len(grids)

	put_cox = np.zeros(n_seq)
	time_cox = np.zeros(n_seq)

	put_bbs = np.zeros(n_seq)
	time_bbs = np.zeros(n_seq)

	put_bbsr = np.zeros(n_seq)
	time_bbsr = np.zeros(n_seq)

	## Main loop ##
	start_time = time.time()

	for k in range(n_seq):

	    # Setting the number of time steps
	    N = grids[k]

	    put_cox[k] = cox(S,vol,r,q,T,K,1,N)
	    put_bbs[k] = bbs(S,vol,r,q,T,K,1,N)
	    put_bbsr[k] = bbsr(S,vol,r,q,T,K,1,N)


	print("--- %.4f seconds ---" % (time.time() - start_time))

	print("\n--- Binomial American Put Prices ---")
	print(put_cox)

	plt.plot(grids, put_cox, '-o', label="Binomial American put price")
	plt.plot(grids, put_bbs, '-o', label="Binomial BS American put price")
	plt.plot(grids, put_bbsr, '-o', 
		label="BBS with Richardson interpolation American put price")

	plt.title("Convergence of American Put price")
	plt.xlabel("Number of iterations")
	plt.ylabel("Binomial model price")
	plt.legend()

	plt.show()















