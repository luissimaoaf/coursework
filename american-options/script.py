import time
import numpy as np

import binomial
import approx
import integral


S0 = 100
T = 0.5
phi = 1

# Different strikes
strikes = [80,90,100,110,120]
n_strikes = len(strikes)

# Different combinations for r, q and vol
params = [[0.07,0.03,0.2],[0.07,0.03,0.4],[0.07,0.0,0.3],[0.03,0.07,0.3]]
n_params = len(params)

# One table for each method
prices = [np.zeros((n_strikes,n_params)) for i in range(9)]
true_prices = np.zeros((n_strikes,n_params))

# Using the "true" prices of Nunes (2009)
true_prices = [[0.219,1.386,4.783,11.098,20],
        [2.689,5.722,10.239,16.181,23.36],
        [1.037,3.123,7.035,12.955,20.717],
        [1.664,4.495,9.25,15.798,23.706]]

true_prices = np.array(true_prices)
true_prices = np.transpose(true_prices)


avg_time = np.zeros(10)

for i in range(n_params):

    r,q,vol = params[i]

    for j in range(n_strikes):
        K = strikes[j]

        # Computing the "true" price
        # We just use the predetermined prices to save computation time
        #start_time = time.time()
        #true_prices[j,i] = binomial.cox(S0,vol,r,q,T,K,phi,15000)
        #avg_time[9] += (time.time() - start_time())

        # Various methods
        # Recursive with 20 steps
        start_time = time.time()
        val, eu, eep, eeb = integral.recursive_pricing_gbm(S0,vol,r,q,T,K,phi,20)
        avg_time[0] += (time.time() - start_time)
        prices[0][j,i] = val

        # Recursive with 100 steps
        start_time = time.time()
        val, eu, eep, eeb = integral.recursive_pricing_gbm(S0,vol,r,q,T,K,phi,100)
        avg_time[1] += (time.time() - start_time)
        prices[1][j,i] = val 
        
        # Cox with 20 steps
        start_time = time.time()
        prices[2][j,i] = binomial.cox(S0,vol,r,q,T,K,phi,20)
        avg_time[2] += (time.time() - start_time)
        
        # Cox with 100 steps
        start_time = time.time()
        prices[3][j,i] = binomial.cox(S0,vol,r,q,T,K,phi,100)
        avg_time[3] += (time.time() - start_time)
        
        # BBS with 20 steps
        start_time = time.time()
        prices[4][j,i] = binomial.bbs(S0,vol,r,q,T,K,phi,20)
        avg_time[4] += (time.time() - start_time)

        # BBS with 100 steps
        start_time = time.time()
        prices[5][j,i] = binomial.bbs(S0,vol,r,q,T,K,phi,100)
        avg_time[5] += (time.time() - start_time)
        
        # BBSR with 20 steps
        start_time = time.time()
        prices[6][j,i] = binomial.bbsr(S0,vol,r,q,T,K,phi,20)
        avg_time[6] += (time.time() - start_time)
        
        # BBSR with 100 steps
        start_time = time.time()
        prices[7][j,i] = binomial.bbsr(S0,vol,r,q,T,K,phi,100)
        avg_time[7] += (time.time() - start_time)
        
        # Analytical approximation
        start_time = time.time()
        prices[8][j,i] = approx.analytic(S0,vol,r,q,T,K,phi)
        avg_time[8] += (time.time() - start_time)


# Computing Mean Absolute Percentage Error
mape = np.array([sum(sum(np.abs(prices[i]-true_prices)/true_prices)) 
    for i in range(9)])
mape = mape/(n_params*n_strikes)

# Average time
avg_time = avg_time/(n_params*n_strikes)

# Printing the results
for table in prices:
    print(np.round(table,3))

print(true_prices)

print('\n')
print(np.round(mape*100,3))
print('\n')
print(np.round(avg_time,4))



