
import time
import numpy as np
import matplotlib.pyplot as plt

import gbm
import cev

import hedging


s0 = 49
mu = 0.13
q = 0
r = 0.05
vol = 0.2
beta = 1.9
T = 0.3846
k = 50

n_sims = 100
n_steps = 20

# Initial simulations for testing
# GBM paths and hedging


start_time = time.time()

gbm_paths = gbm.simulate(s0,mu,vol,T,n_steps,n_sims)
gbm_stoploss_costs = hedging.stoploss(k,r,T,gbm_paths)
gbm_deltahedge_costs = hedging.delta_bsm(q,r,vol,k,T,gbm_paths)

print("--- Run time: %.4f seconds ---" % (time.time() - start_time))
print("\n Average cost of stop-loss hedging: %.4f" % gbm_stoploss_costs.mean())
print("\n Average cost of delta hedging: %.4f" % gbm_deltahedge_costs.mean())

# CEV paths and hedging

start_time = time.time()

cev_paths = cev.simulate(s0,mu,vol,beta,T,n_steps,n_sims)
cev_paths = np.array([path[::5] for path in cev_paths])

cev_stoploss_costs = hedging.stoploss(k,r,T,cev_paths)
cev_deltahedge_costs = hedging.delta_cev(q,r,vol,beta,k,T,cev_paths)

print("--- Run time: %.4f seconds ---" % (time.time() - start_time))
print("\n Average cost of stop-loss hedging: %.4f" % cev_stoploss_costs.mean())
print("\n Average cost of delta hedging: %.4f" % cev_deltahedge_costs.mean())

# Final plots

plt.hist(gbm_stoploss_costs, bins='auto')
plt.show()

plt.hist(gbm_deltahedge_costs, bins='auto')
plt.show()

plt.hist(cev_stoploss_costs, bins='auto')
plt.hist(cev_deltahedge_costs, bins='auto')
plt.show()






