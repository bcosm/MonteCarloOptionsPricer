# Hybrid Monte Carlo Options Pricer

[![CI](https://github.com/bcosm/MonteCarloOptionsPricer-new/actions/workflows/ci.yml/badge.svg)](https://github.com/bcosm/MonteCarloOptionsPricer-new/actions/workflows/ci.yml)

This repository contains an Monte Carlo–based options pricer that implements multiple American-style option pricing methods under rough volatility dynamics. The goal is to compare and combine various algorithmic techniques for accurate pricing under realistic market conditions. This project also integrates a Bayesian Neural Network for post-processing or meta-modeling the generated paths and prices.

---

## Key Features

1. **Modular Design:**  
   Each pricing method is defined in its own module, making it easy to plug in or remove pricing algorithms.

2. **Hybrid Monte Carlo Engine:**  
   Combines standard Monte Carlo path generation with rough volatility processes and multiple early-exercise estimators.

3. **OpenMP Parallelization:**  
   Speeds up large batch simulations, especially for American-style contract pricing.

4. **Optional CUDA Support:**  
   The Bayesian Neural Network (BNN) can run on GPU for faster training and inference.

---

## Pricing Methods

This project provides several distinct pricing algorithms for American-style options, each focusing on a different aspect of numerical efficiency and accuracy:

1. **Asymptotic Analysis**  
   - Uses boundary approximations for early-exercise.
   - Useful for quickly estimating exercise boundaries and payoffs.

2. **Branching Processes**  
   - Approximates upper and lower bounds by exploring a tree of possible future states.
   - Balances computational complexity with flexibility in payoff structures.

3. **Least Squares Monte Carlo (LSM)**  
   - A regression-based approach (Longstaff–Schwartz).
   - Regresses continuation values at each step to handle path-dependent exercise decisions.

4. **Martingale Optimization**  
   - Relies on duality principles to reduce variance in American option pricing.
   - Often offers tighter bounds on the true option price.

These methods are guided by techniques outlined in the work of **Keller, etc.**, as described in [the reference paper](#reference-paper).

---

## Bayesian Neural Network

A Torch-based Bayesian Meta Model is included:

- **MetaModeling:** Predicts final option prices or adjustments using MC Dropout for uncertainty estimation.  
- **Multiple Forward Passes:** Helps quantify the prediction variance around the option prices.  
- **Flexible Architectures:** Allows custom layer definitions, normalizing flows, and mixture density heads.

---

## Rough Volatility Model

The Rough Volatility approach:
- **Fractional Gaussian Noise:** Captures long-memory effects of volatility (commonly observed in market data).  
- **FFT Techniques:** Speeds up fractional Brownian motion generation (rBergomi style).  
- **Parameter Estimation:** Automatically estimates Hurst exponent, volatility of volatility, and correlation from historical data.

---

## Reference Paper

Methods for pricing American options in this project draw from:
- **M. Caflisch, W. Morokoff, T. Ray, and M. Giles** in  
  *Keller Meeting on American Option Pricing*, 2005.  
  [[Link]](https://www.math.ucla.edu/~caflisch/Pubs/Pubs2005/KellerMeet2005.pdf)
