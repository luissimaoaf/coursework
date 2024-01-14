
"""
misspecification.py

This script performs a dynamic hedging experiment comparing hedging results when
the model is misspecificed.
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
true_vol = 0.25
true_beta = 1
k = 50
T = 0.3846

# Different parameters for testing misspecification effects
# beta=2 includes misspecifying the model as a GBM
# For simplificty I assume we have the same ammount of vols and betas
betas = [-1,2,3]
vols = [0.15, 0.35, 0.5]
n_params = len(betas)

################################################################################
# Simulation parameters and storage initialization
################################################################################

# Total number of simulations
n_sims = 100000

# We test for model misspecification costs at a weekly hedging interval
n_steps = 20

# To store hedging results
# Only delta hedging requires model specification
miss_results = np.array(np.zeros((2,n_params)),dtype='object')

# To store the average costs
miss_costs = np.zeros((2,n_params))

# To store the standard deviations
miss_std = np.zeros((2,n_params))

################################################################################
# Beggining the simulations
################################################################################

start_time = time.time()

print("Sampling %s paths with %s time steps..." % (n_sims,n_steps))

# Sample paths of the "true" model (CEV with sigma=0.25 and beta=1)
true_paths = cev.simulate(s0,mu,true_vol,true_beta,T,n_steps,n_sims)

sampling_time = time.time() - start_time
print("Done in %.2f seconds.\n" % sampling_time)

sim_start = time.time()

print("Starting %s simulations..." % n_sims)

# Well-specified delta hedging
truedelta_results = hedging.delta_cev(q,r,true_vol,true_beta,k,T,true_paths)
truedelta_costs = truedelta_results.mean()
truedelta_std = truedelta_results.std()

# Model independent hedges
naked_positions = [np.exp(-r*T)*max(s-k,0) for s in true_paths[:,-1]]
covered_positions = [(s0 - np.exp(-r*T)*min(k,s)) for s in true_paths[:,-1]]
stoploss_hedges = hedging.stoploss(k,r,T,true_paths)

naked_costs = np.mean(naked_positions)
naked_std = np.std(naked_positions)

covered_costs = np.mean(covered_positions)
covered_std = np.std(covered_positions)

stoploss_costs = stoploss_hedges.mean()
stoploss_std = stoploss_hedges.std()

print("Starting misspecified model simulations...")
# Delta hedging
for i in range(n_params):
    
    miss_vol = vols[i]
    miss_beta = betas[i]

    print('Mis-specified vol to %.2f...' % miss_vol)

    # Misspecified vol delta hedging 
    miss_results[0,i] = hedging.delta_cev(q,r,miss_vol,true_beta,k,T,true_paths)
    miss_costs[0,i] = miss_results[0,i].mean()
    miss_std[0,i] = miss_results[0,i].std()

    print('Mis-specified beta to %d...' % miss_beta)

    # Misspecified beta delta hedging
    miss_results[1,i] = hedging.delta_cev(q,r,true_vol,miss_beta,k,T,true_paths)
    miss_costs[1,i] = miss_results[1,i].mean()
    miss_std[1,i] = miss_results[1,i].std() 

print("Done!\n")
print('Simulation time: %.4f seconds' % (time.time() - sim_start))
print('Total running time: %.4f seconds\n' % (time.time() - start_time))

print('Creating cost histograms...')

fig, ax = plt.subplots(2,2)
fig.suptitle('Well-specified/model independent hedging')

ax[0,0].hist(truedelta_results,bins='auto',density=True)
ax[0,0].set_title('Delta hedging')

ax[0,1].hist(stoploss_hedges,bins='auto',density=True)
ax[0,1].set_title('Stop loss')

ax[1,0].hist(naked_positions,bins='auto',density=True)
ax[1,0].set_title('Naked call')

ax[1,1].hist(covered_positions,bins='auto',density=True)
ax[1,1].set_title('Covered call')

plt.tight_layout()
fig.savefig('miss_results/true_costs.png')
plt.close(fig)

fig, ax = plt.subplots(2,n_params)
fig.suptitle('Miss-specified hedging')

for i in range(n_params):
    
    ax[0,i].hist(miss_results[0,i],bins='auto',density=True)
    ax[0,i].set_title('vol=%.2f' % vols[i])
    
    ax[1,i].hist(miss_results[1,i],bins='auto',density=True)
    ax[1,i].set_title('beta=%d' % betas[i])

plt.tight_layout()
fig.savefig('miss_results/miss_costs.png')
plt.close(fig)

# Saving the results as CSV files
print('Saving results as CSV files...')

true_costs = [truedelta_costs, stoploss_costs, naked_costs, covered_costs]
true_std = [truedelta_std, stoploss_std, naked_std, covered_std]

np.savetxt('miss_results/true_costs.csv', true_costs, delimiter=',')
np.savetxt('miss_results/true_std.csv', true_std, delimiter=',')
np.savetxt('miss_results/miss_costs.csv', miss_costs, delimiter=',')
np.savetxt('miss_results/miss_std.csv', miss_std, delimiter=',')

print('Done!')




