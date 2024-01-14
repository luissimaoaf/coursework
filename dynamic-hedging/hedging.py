"""
hedging.py
Contains different hedging strategies for European call options.
Each strategy takes given parameters as inputs, as well as a numpy array of
stock paths to simulate the strategy over.
"""


import numpy as np
import time
import greeks


def stoploss(k, r, T, stock_paths):
    """
    hedging.stoploss
    Stop loss hedging strategy.
    """

    [n_sims, n_hedges] = stock_paths.shape
    dt = T/(n_hedges-1)
    discount_factors = [np.exp(-r*dt*k) for k in range(n_hedges)]
    total_costs = np.zeros(n_sims)

    for j in range(n_sims):

        stock_path = stock_paths[j]
        balance = np.zeros(n_hedges)
        covered = 0

        for i in range(n_hedges):
        
            if stock_path[i] > k and covered == 0:
                # Buy stock when price is above strike
                covered = 1
                balance[i] = -stock_path[i]

            elif stock_path[i] < k and covered == 1:
                # Sell stock when price is bellow strike
                covered = 0
                balance[i] = stock_path[i]
        
        if stock_path[-1] > k:
            # Option is exercized and the stock is sold at the strike
            balance[-1] += k

        total_costs[j] = -np.dot(discount_factors, balance)

    return total_costs


def delta_bsm(q,r,vol,k,T,stock_paths):
    """
    hedging.delta_bsm 
    Delta hedging under the GBM assumption.
    """

    [n_sims, n_hedges] = stock_paths.shape
    dt = T/(n_hedges-1)
    discount_factors = [np.exp(-r*dt*k) for k in range(n_hedges)]
    total_costs = np.zeros(n_sims)

    for j in range(n_sims):

        stock_path = stock_paths[j]
        balance = np.zeros(n_hedges)
        shares = 0

        for i in range(n_hedges-1):
            # We update the value of the option delta
            tau = T - i*dt 
            delta = greeks.delta_bsm(stock_path[i],q,r,vol,k,tau,-1)
            # Adjust the number of shares held
            balance[i] = (shares - delta) * stock_path[i]
            shares = delta
        
        if stock_path[-1] >= k:
            # Option is exercised
            balance[-1] += (delta - 1)*stock_path[-1] + k
        else:
            balance[-1] += delta * stock_path[-1]
        
        total_costs[j] = -np.dot(discount_factors, balance)

    return total_costs


def delta_cev(q,r,sigma,beta,k,T,stock_paths):
    """
    hedging.delta_cev
    Delta hedging under the CEV assumption.
    """

    [n_sims, n_hedges] = stock_paths.shape
    dt = T/(n_hedges-1)
    discount_factors = [np.exp(-r*dt*k) for k in range(n_hedges)]
    total_costs = np.zeros(n_sims)

    for j in range(n_sims):

        stock_path = stock_paths[j]
        balance = np.zeros(n_hedges)
        shares = 0

        for i in range(n_hedges-1):
            # We update the value of the option delta
            tau = T - i*dt 
            delta = greeks.delta_cev(stock_path[i],q,r,sigma,beta,k,tau,-1)
            # Adjust the number of shares held
            balance[i] = (shares - delta) * stock_path[i]
            shares = delta
        
        if stock_path[-1] >= k:
            # Option is exercised
            balance[-1] += (delta - 1)*stock_path[-1] + k
        else:
            balance[-1] += delta * stock_path[-1]
        
        total_costs[j] = -np.dot(discount_factors, balance)

    return total_costs






