hedging

This module allows us to perform hedging simulations of different strategies,
including stop loss, covered calls, naked calls and delta hedging under the GBM
and CEV assumptions.

- gbm.py
    Script for simulating Geometric Brownian Motion paths.

- cev.py
    Script for simulating paths of the CEV process using a first order Euler
    discretization.

- greeks.py
    Script for computing the delta of the option under the different
    assumptions.

- hedging.py
    Script implementing the different hedging strategies. Each works in the same
    way, taking in a given set of stock paths. This allows us to also test for
    model misspecification.

- hedging_script.py
    Test script for showcasing the general behavior.

- rebalancing_times.py
    Experiment about the effect of increasing rebalancing times. Could account
    for trading costs in the future.

- misspecification.py
    Experiment about the effect of model misspecification.
