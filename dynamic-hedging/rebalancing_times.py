"""
rebalancing_times.py

This script performs a simulation comparison between different rebalancing times
for different hedging strategies (naked, covered, stop loss and delta).
Saves histogram plots of cost distributions and CSV files with the corresponding 
mean and standard deviation values.
"""

import time
import numpy as np
import matplotlib.pyplot as plt

import gbm
import cev
import hedging


################################################################################
# Model parameters
################################################################################

s0 = 49
mu = 0.13
q = 0
r = 0.05
vol = 0.25
T = 0.3846
k = 50

# Different betas for testing
betas = [-1, 0, 1, 2.2]
n_betas = len(betas)

################################################################################
# Simulation parameters and storage initialization
################################################################################

# Total number of simulations
# With the 5 different paths (4 beta values + GBM) we get the 100000 stock paths
n_sims = 20000

# Different time steps we are experimenting with
n_steps_experiment = [6, 20, 50, 100, 200]
n_experiments = len(n_steps_experiment)
max_steps = np.max(n_steps_experiment)

# To store the average costs
gbm_costs = np.zeros(n_experiments)
cev_costs = np.zeros((n_experiments,n_betas))
stoploss_costs = np.zeros((n_experiments,n_betas+1))
covered_costs = np.zeros((n_experiments,n_betas+1))
naked_costs = np.zeros((n_experiments,n_betas+1))

# To store the standard deviations
gbm_std = np.zeros(n_experiments)
stoploss_std = np.zeros((n_experiments,n_betas+1))
covered_std = np.zeros((n_experiments,n_betas+1))
naked_std = np.zeros((n_experiments,n_betas+1))
cev_std = np.zeros((n_experiments,n_betas))

################################################################################
# Beggining the simulations
################################################################################

start_time = time.time()

# We sample the paths first to use the same paths for the computations and to
# save time, and subsample as needed
# We simulate max_steps+1 to ensure divisibility without much error
gbm_paths = gbm.simulate(s0,mu,vol,T,max_steps+1,n_sims)
cev_paths = [cev.simulate(s0,mu,vol,beta,T,max_steps+1,n_sims) 
            for beta in betas]

sampling_time = time.time() - start_time

sim_start = time.time()

print("Starting %s simulations...\n" % n_sims)

for i in range(n_experiments):

    n_steps = n_steps_experiment[i]
    print("Number of hedges: %s\n" % n_steps)
    window = int(max_steps/n_steps)
    
    # Getting the subsamples
    gbm_subsample = gbm_paths[:,::window]
    cev_subsamples = [cev_path[:,::window] for cev_path in cev_paths]
    
    # Creating figure with subplots for GBM histograms
    fig, ax = plt.subplots(2,2)
    fig.suptitle('Hedging costs under GBM and %d hedges' % n_steps)

    # GBM costs - naked
    # In a naked short call position, the costs are just the payoffs of the
    # option
    results = [np.exp(-r*T)*max(s-k,0) for s in gbm_subsample[:,-1]]
    naked_costs[i,0] = np.mean(results)
    naked_std[i,0] = np.std(results)
    ax[0,0].hist(results,bins='auto',density=True)
    ax[0,0].set_title('Naked position')

    # GBM costs - covered
    # In a covered short call position, we pay S0 initially and get back K at
    # most at the end of the period
    results = [s0-np.exp(-r*T)*min(k,s) for s in gbm_subsample[:,-1]]
    covered_costs[i,0] = np.mean(results)
    covered_std[i,0] = np.std(results)
    ax[0,1].hist(results,bins='auto',density=True)
    ax[0,1].set_title('Covered position')
    
    # GBM costs - stop loss
    results = hedging.stoploss(k,r,T,gbm_subsample)
    stoploss_costs[i,0] = results.mean()
    stoploss_std[i,0] = results.std()
    ax[1,0].hist(results,bins='auto',density=True)
    ax[1,0].set_title('Stop loss')

    # Dynamic delta hedging costs under GBM
    results = hedging.delta_bsm(q,r,vol,k,T,gbm_subsample)
    gbm_costs[i] = results.mean()
    gbm_std[i] = results.std()
    ax[1,1].hist(results,bins='auto',density=True)
    ax[1,1].set_title('Dynamic hedging')

    # Saving the figure as png
    plt.tight_layout()
    fig.savefig('hedge_results/gbm_costs_%d_hedges.png' % n_steps)
    plt.close(fig)

    for j in range(n_betas):

        # Creating figure for current beta
        # Unreadable with all of them together
        fig, ax = plt.subplots(2,2)
        fig.suptitle('Hedging costs under CEV (beta=%.1f) and %d hedges' %
                (betas[j], n_steps))

        # Naked
        results = [np.exp(-r*T)*max(s-k,0) for s in cev_subsamples[j][:,-1]]
        naked_costs[i,j+1] = np.mean(results)
        naked_std[i,j+1] = np.std(results)
        ax[0,0].hist(results,bins='auto',density=True)
        ax[0,0].set_title('Naked position')

        # Covered
        results = [(s0-np.exp(-r*T)*min(k,s)) for s in cev_subsamples[j][:,-1]]
        covered_costs[i,j+1] = np.mean(results)
        covered_std[i,j+1] = np.std(results)
        ax[0,1].hist(results,bins='auto',density=True)
        ax[0,1].set_title('Covered position')

        # Stop loss
        results = hedging.stoploss(k,r,T,cev_subsamples[j])
        stoploss_costs[i,j+1] = results.mean()
        stoploss_std[i,j+1] = results.std()
        ax[1,0].hist(results,bins='auto',density=True)
        ax[1,0].set_title('Stop loss')

        # Dynamic
        results = hedging.delta_cev(q,r,vol,betas[j],k,T,cev_subsamples[j])
        cev_costs[i,j] = results.mean()
        cev_std[i,j] = results.std()
        ax[1,1].hist(results,bins='auto',density=True)
        ax[1,1].set_title('Dynamic hedging')

        # Saving the figure as png
        plt.tight_layout()
        fig.savefig('hedge_results/cev%d_%d_hedges.png' % (j,n_steps))
        plt.close(fig)

print('Total running time: %.4f seconds' % (time.time() - start_time))
print('Sampling time: %.4f seconds' % (sampling_time))
print('Simulation time: %.4f seconds\n' % (time.time() - sim_start))


# Saving the results as CSV files
print('Saving results as CSV files...')
np.savetxt('hedge_results/naked_costs.csv', naked_costs, delimiter=',')
np.savetxt('hedge_results/covered_costs.csv', covered_costs, delimiter=',')
np.savetxt('hedge_results/stoploss_costs.csv', stoploss_costs, delimiter=',')
np.savetxt('hedge_results/cev_costs.csv', cev_costs, delimiter=',')
np.savetxt('hedge_results/gbm_costs.csv', gbm_costs, delimiter=',')

np.savetxt('hedge_results/naked_std.csv', naked_std, delimiter=',')
np.savetxt('hedge_results/covered_std.csv', covered_std, delimiter=',')
np.savetxt('hedge_results/stoploss_std.csv', stoploss_std, delimiter=',')
np.savetxt('hedge_results/cev_std.csv', cev_std, delimiter=',')
np.savetxt('hedge_results/gbm_std.csv', gbm_std, delimiter=',')

print('Done!')





