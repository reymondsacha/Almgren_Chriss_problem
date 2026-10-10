import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from derivation import DiscreteAlmgrenChriss
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Define market and execution parameters
X_total: float =  5e4       # Initial inventory
T_horizon: float = 1.0          # Time horizon (e.g., 1 day)
N_steps: int = 50               # Number of trading intervals
sigma_vol: float = 0.3          # Arithmetic volatility
eta_temp: float = 2e-5          # Temporary market impact
gamma_perm: float = 1e-5       # Permanent market impact



def plot_single_trajectory(time_grid: np.ndarray, lambda_val: float):        
    # Instantiate the solver
    ac_model = DiscreteAlmgrenChriss(
       X=X_total,
       T=T_horizon,
       N=N_steps,
       sigma=0.58,
       eta=1.7e-9,
       gamma=1.7e-10,
       lambda_penalty=lambda_val
    )

    # Compute the optimal trajectory
    optimal_x: np.ndarray = ac_model.solve_trajectory()
    

    # Visualize the trajectory
    plt.figure(figsize=(10, 6))
    plt.plot(time_grid, optimal_x, marker='.', linestyle='-', color='b', label=rf'$\lambda = {lambda_val}$')
    plt.title("Discrete Almgren-Chriss Optimal Execution Trajectory")
    plt.xlabel("Time ($t$)")
    plt.ylabel("Inventory ($x_k$)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Verify boundary conditions numerically
    print(f"Initial Inventory (x_0): {optimal_x[0]}")
    print(f"Final Inventory (x_N): {optimal_x[-1]}")

time_grid: np.ndarray = np.linspace(0, T_horizon, N_steps + 1)
#plot_single_trajectory(time_grid, lambda_val = 1e-6)






def lambda_boundary_behavior(lambdas):
    """
    We aim to compute a matrix that represents 50 trajectories following the Almgren-Chriss model
    with λ spanning from 10e-6 to 10. What we expect to observe is that as λ -> 0, the curve becomes
    a straight line and follows the TWAP whereas when λ -> +∞, the execution becomes more convex and
    front-loaded with a first step liquidation representing a high stake of the total inital portfolio.
    """

    trajectory = np.zeros(shape = (len(lambdas), N_steps +1))

    for i in range(len(lambdas)):
        ac_model = DiscreteAlmgrenChriss(
    X=X_total,
    T=T_horizon,
    N=N_steps,
    sigma=sigma_vol,
    eta=eta_temp,
    gamma=gamma_perm,
    lambda_penalty=lambdas[i]
)
        x = ac_model.solve_trajectory()
        trajectory[i] = x
        
    return trajectory





def plot_lambda_boundaries(time_grid: np.ndarray, trajectory: np.ndarray, lambdas: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate a vectorized color map for the trajectories
    colors = cm.viridis(np.linspace(0, 1, len(lambdas)))
    ax.set_prop_cycle(color=colors)
    
    # Plot all trajectories simultaneously (transpose aligns N+1 rows with the time grid)
    ax.plot(time_grid, trajectory.T, alpha=0.8)
    
    # Configure the logarithmic colorbar
    norm = mcolors.LogNorm(vmin=lambdas.min(), vmax=lambdas.max())
    scalar_mappable = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    scalar_mappable.set_array(np.array([]))
    cbar = fig.colorbar(scalar_mappable, ax=ax)
    cbar.set_label(r'Risk Aversion Penalty ($\lambda$)')
    
    ax.set_title("Optimal Execution Trajectories across Risk Aversion Levels")
    ax.set_xlabel("Time ($t$)")
    ax.set_ylabel("Inventory ($x_k$)")
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("Optimal_Execution_Trajectories_across_Risk_Aversion_Levels.png")
    plt.show()


time_grid = np.linspace(0, T_horizon, N_steps + 1)
lambdas = np.logspace(-6, 1, num=50, base=10)

trajectory = lambda_boundary_behavior(lambdas) 
#plot_lambda_boundaries(time_grid, trajectory, lambdas)





def numerical_verification():
    """
    We want to verify if our assumptions regarding the behavior of the Almgren-Chriss trajectory
    depending on lambda are right.
    We first calculate the infinite norm between the TWAP trajectory and the Almgren-Chriss for λ = 10e-6.
    Then we compute the fraction of stocks liquidated on the first step with λ = 10.
    """
    
    #1st part for λ = 10e-6
    ac_model = DiscreteAlmgrenChriss(
    X=X_total,
    T=T_horizon,
    N=N_steps,
    sigma=sigma_vol,
    eta=eta_temp,
    gamma=gamma_perm,
    lambda_penalty=10**(-6)
    )

    x = ac_model.solve_trajectory()

    x_TWAP = X_total * (1 - np.arange(0, N_steps+1) / N_steps)

    L_inf = np.max(np.abs(x_TWAP - x))


    #2nd part for λ = 10
    ac_model_2 = DiscreteAlmgrenChriss(
    X=X_total,
    T=T_horizon,
    N=N_steps,
    sigma=sigma_vol,
    eta=eta_temp,
    gamma=gamma_perm,
    lambda_penalty=10
    )

    x = ac_model_2.solve_trajectory()

    ratio = (X_total - x[1]) / X_total



    return L_inf, ratio

L_inf, ratio = numerical_verification()
#print(f"L_inf the infinite distance between the Almgren-Chriss trajectory for λ = 10e-6 and the TWAP model is {L_inf: .4f}")

#print(f"The fraction of stocks liquidated in the first step for λ = 10 is {ratio: .4f}")




from data_downloader import MicrostructureDownloader
from dataset_builder import build_ml_dataset
from features import Market_State_Engineer, Execution_Parameter_Regressor
from dynamic_execution import DynamicAlmgrenChriss

def dynamic_execution(lam: int, plot:bool = True):
    """
    Orchestrates the full ML pipeline: data ingestion, feature engineering,
    Ridge model training, and dynamic execution trajectory simulation.
    """
    # 1. Ingest Data
    downloader = MicrostructureDownloader(data_dir="data/raw")
    # Using a 10-day window within the last 60 days to comply with yfinance limits
    df = downloader.fetch_intraday(ticker="AAPL", start_date="2026-09-13", end_date="2026-09-27", interval="5m")
    

    if df.empty:
        print("Data fetch failed. Cannot run Phase 3.")
        return

    # 2. Engineer Features and Targets
    window_size = 6        # 30-minute trailing window (6 * 5m)
    forward_periods = 3    # 15-minute forward prediction window (3 * 5m)
    
    engineer = Market_State_Engineer(window=window_size)
    X, y_sigma, y_eta = build_ml_dataset(df, engineer, m_forward_periods=forward_periods)

    # 3. Train the Ridge Regressors
    alphas = np.logspace(-4, 2, 20)
    sigma_model = Execution_Parameter_Regressor(alphas=alphas)
    eta_model = Execution_Parameter_Regressor(alphas=alphas)
    
    #print("Training Volatility Regressor...")
    sigma_model.fit(X, y_sigma)
    
    #print("Training Liquidity Regressor...")
    eta_model.fit(X, y_eta)

    # 4. Simulate Live Execution
    # We slice the last N_steps from our feature matrix to simulate a live trading day
    X_live_simulation = X.iloc[-N_steps:]

    gamma_perm_calibrated = np.mean(np.exp(y_eta)) * 0.1

    mean_y_sigma = np.mean(np.exp(y_sigma))
    mean_y_eta = np.mean(np.exp(y_eta))
    
    dynamic_engine = DynamicAlmgrenChriss(
        X=X_total, 
        T=T_horizon, 
        N=N_steps, 
        gamma=gamma_perm_calibrated, 
        lambda_penalty=lam  # Example risk aversion
    )
    
    dynamic_trajectory = dynamic_engine.execute_dynamic_trajectory(
        feature_matrix=X_live_simulation, 
        sigma_model=sigma_model, 
        eta_model=eta_model,
    )

    # 5. Plot the Comparison
    # We instantiate the static Phase 1 model using the historical mean of the targets
    static_model = DiscreteAlmgrenChriss(
        X=X_total, 
        T=T_horizon, 
        N=N_steps, 
        sigma=mean_y_sigma, 
        eta=mean_y_eta, 
        gamma=gamma_perm_calibrated, 
        lambda_penalty=lam
    )
    static_trajectory = static_model.solve_trajectory()

    if plot:
        time_grid = np.linspace(0, T_horizon, N_steps + 1)
    
        plt.figure(figsize=(10, 6))
        plt.plot(time_grid, static_trajectory, linestyle='--', color='gray', label='Static (Phase 1)')
        plt.plot(time_grid, dynamic_trajectory, marker='.', linestyle='-', color='blue', label='Dynamic MPC (Phase 3)')
        plt.title("Execution Trajectory: Static vs. Dynamic Machine Learning")
        plt.xlabel("Time ($t$)")
        plt.ylabel("Inventory ($x_k$)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    return {
        "df": df,
        "feature_matrix": X,
        "X_total": X_total,
        "N_steps": N_steps,
        "T_horizon": T_horizon,
        "static_trajectory": static_trajectory,
        "dynamic_engine": dynamic_engine,
        "sigma_model": sigma_model,
        "eta_model": eta_model,
        "y_eta": y_eta
    }

# Execute Phase 3
dynamic_execution(lam=2e-8)

