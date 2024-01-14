The binomial module includes scripts implementing the Binomial Model of Cox et
al., the Binomial Black-Scholes model and the BBS model with Richardson
extrapolation.

Files: 

- bsm_formula.py
    Includes helper functions for the cumulative density function of the
    standard normal distribution and the Black-Scholes formula to be used in the
    main scripts.

- cox-bsm.py
    Implements the binomial model of Cox et al. and performs a convergence
    experiment with increasing time steps.
    Outputs a plot of the log-absolute errors for both the call and put options,
    saved locally in a PNG file.

- bbs-bsm.py
    Implements the BBS and BBSR methods and performs a convergence experiment
    with increasing time steps.
    Outputs two plots (for calls and puts) comparing the convergence of both
    methods, saved locally as PNG files.
