# Discrete Almgren-Chriss Optimal Execution & ML-Driven MPC

A quantitative framework for optimal trade execution, implementing a strictly discrete formulation of the Almgren-Chriss model extended with a Machine Learning-driven Model Predictive Control (MPC) engine.

This repository models the liquidation of a portfolio under permanent and temporary market impact, utilizing a vectorized $O(N)$ numerical solver for the unconstrained quadratic optimization problem. It bridges static theoretical execution schedules with dynamic, non-linear market microstructure adaptations using regularized supervised learning.

## System Architecture

* **Analytical & Numerical Core**: Solves the mean-variance objective function $U(\mathbf{x}) = \mathbb{E}[C(\mathbf{x})] + \lambda \mathbb{V}[C(\mathbf{x})]$ by isolating the interior points into a strictly convex multi-dimensional quadratic form $\frac{1}{2}\mathbf{x}^T \mathbf{A} \mathbf{x} - \mathbf{b}^T \mathbf{x}$. The resulting symmetric positive-definite tridiagonal system is inverted using `scipy.linalg.solve_banded` for maximum computational efficiency.
* **Vectorized Monte Carlo Engine**: An event-driven simulation framework generating 10,000+ simultaneous arithmetic random walk paths. It computes empirical execution shortfalls and variance exactly, validating the continuous theoretical Efficient Frontier against discrete stochastic realizations without iterative loops.
* **Dynamic MPC & Machine Learning**: Replaces static impact parameters by predicting localized volatility ($\sigma_{t+1}$) and temporary impact ($\eta_{t+1}$) using a Ridge ($L_2$ regularized) regressor trained on high-frequency order book features (Garman-Klass volatility, Order Flow Imbalance).
* **Empirical Bootstrapper**: A historical grading engine that evaluates the execution trajectories against actual market data (e.g., AAPL), accounting for cumulative permanent impact and localized liquidity scoring to isolate the execution penalty from intraday Brownian noise.

## Mathematical Formulation

The fundamental price process follows an arithmetic random walk with permanent market impact $\gamma$:
$$S_{k} = S_{k-1} + \sigma \sqrt{\tau} \xi_{k} - \gamma v_{k}$$

The executed price incorporates a temporary market impact penalty $\eta$:
$$\overline{S}_{k} = S_{k-1} - \eta \frac{v_{k}}{\tau}$$

The numerical solver evaluates the discrete second-order difference equation, strictly converging to the continuous-time ODE $x''(t) - \tilde{\kappa}^2 x(t) = 0$ as $\tau \to 0$.

## Repository Structure

    ├── data/                  # Historical price and volume data for MPC calibration
    ├── notebooks/             # Jupyter notebooks containing mathematical proofs and EDA
    │   ├── validation.ipynb   # Monte Carlo Efficient Frontier validation
    │   └── ml_calibration.ipynb # Ridge regression feature engineering and tuning
    ├── src/
    │   ├── solver.py          # Vectorized tridiagonal matrix solver (scipy.linalg.solve_banded)
    │   ├── simulation.py      # Vectorized Monte Carlo price path generation
    │   ├── mpc_engine.py      # ML-driven dynamic execution loop
    │   └── evaluation.py      # Empirical bootstrapper and IS distribution grading
    ├── docs/
    │   └── theory.pdf         # Comprehensive mathematical documentation and LaTeX proofs
    ├── pyproject.toml         # Poetry dependency management
    └── README.md

## Core Results

1. **Theoretical Validation**: The discrete numerical solver aligns with the continuous-time hyperbolic analytical limit to within 0.11% discretization error.
2. **Efficient Frontier Convergence**: Monte Carlo empirical estimators perfectly map onto the continuous theoretical boundary across the logarithmic $\lambda$ grid.
3. **Liquidity Stress-Testing**: By scaling historical impact parameters by $10^4$ to suppress Brownian noise, the empirical bootstrapper validates that the dynamic MPC solver successfully routes volume away from localized liquidity droughts, reverting to the strict variance-minimizing schedule under infinite constraints.

## Installation & Usage

This project uses Poetry for dependency management.

    # Clone the repository
    git clone https://github.com/username/almgren-chriss-mpc.git
    cd almgren-chriss-mpc
    
    # Install dependencies
    poetry install
    
    # Run the core analytical solver
    poetry run python src/solver.py
    
    # Execute the Monte Carlo simulation suite
    poetry run python src/simulation.py

## Dependencies

* `numpy`: Core array operations and vectorized calculations.
* `scipy`: Banded matrix inversion (`scipy.linalg.solve_banded`).
* `pandas`: Time-series feature engineering.
* `scikit-learn`: Ridge regression and time-series cross-validation.
* `matplotlib` / `seaborn`: Visualization of execution trajectories and Efficient Frontiers.