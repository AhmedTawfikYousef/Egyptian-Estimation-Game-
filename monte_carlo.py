import numpy as np
import matplotlib.pyplot as plt

def simulate_price_paths(S0, mu, sigma, T, dt, n_paths):
    """
    Simulates stock price paths using Geometric Brownian Motion.

    Args:
        S0 (float): Initial stock price.
        mu (float): Expected annual return.
        sigma (float): Annual volatility.
        T (float): Time horizon in years.
        dt (float): Time step in years.
        n_paths (int): Number of paths to simulate.

    Returns:
        np.ndarray: A 2D numpy array of simulated price paths.
    """
    n_steps = int(T / dt)
    # Generate random numbers for all paths and steps at once
    z = np.random.normal(size=(n_paths, n_steps))
    # Calculate the price increments
    price_increments = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    # Stack the initial price to the increments
    paths = np.hstack([np.full((n_paths, 1), S0), price_increments])
    # Calculate the cumulative product to get the paths
    paths = np.cumprod(paths, axis=1)
    return paths[:, 1:]


def run_trade_simulation(S0, mu, sigma, T, dt, n_paths, stop_loss, take_profit):
    """
    Runs the Monte Carlo simulation for a trading strategy.

    Args:
        S0 (float): Initial stock price.
        mu (float): Expected annual return.
        sigma (float): Annual volatility.
        T (float): Time horizon in years.
        dt (float): Time step in years.
        n_paths (int): Number of paths to simulate.
        stop_loss (float): Stop-loss price.
        take_profit (float): Take-profit price.

    Returns:
        tuple: A tuple containing probabilities, expected return, and the simulated paths.
    """
    n_steps = int(T / dt)
    paths = simulate_price_paths(S0, mu, sigma, T, dt, n_paths)

    hit_stop_matrix = paths <= stop_loss
    hit_target_matrix = paths >= take_profit

    first_hit_stop_idx = np.argmax(hit_stop_matrix, axis=1)
    first_hit_target_idx = np.argmax(hit_target_matrix, axis=1)

    first_hit_stop_idx[np.all(~hit_stop_matrix, axis=1)] = n_steps
    first_hit_target_idx[np.all(~hit_target_matrix, axis=1)] = n_steps

    stop_first = first_hit_stop_idx <= first_hit_target_idx
    target_first = first_hit_target_idx < first_hit_stop_idx

    profits = np.zeros(n_paths)
    profits[stop_first] = stop_loss - S0
    profits[target_first] = take_profit - S0
    neither_hit = (first_hit_stop_idx == n_steps) & (first_hit_target_idx == n_steps)
    profits[neither_hit] = paths[neither_hit, -1] - S0

    stop_prob = np.sum(stop_first) / n_paths
    target_prob = np.sum(target_first) / n_paths
    expected_return = np.mean(profits)

    return stop_prob, target_prob, expected_return, paths


def calculate_position_size(capital, risk_per_trade, stop_loss, S0):
    """
    Calculates the suggested position size based on stop-loss.

    Args:
        capital (float): Total capital.
        risk_per_trade (float): Risk per trade as a percentage of capital.
        stop_loss (float): Stop-loss price.
        S0 (float): Initial stock price.

    Returns:
        float: The suggested position size.
    """
    max_loss_per_trade = capital * risk_per_trade
    potential_loss_per_share = S0 - stop_loss
    if potential_loss_per_share <= 0:
        return 0  # Avoid division by zero or negative loss
    return max_loss_per_trade / potential_loss_per_share


def plot_simulation(paths, stop_loss, take_profit):
    """
    Plots the simulated stock price paths.

    Args:
        paths (np.ndarray): The simulated price paths.
        stop_loss (float): Stop-loss price.
        take_profit (float): Take-profit price.
    """
    for i in range(min(100, len(paths))):
        plt.plot(paths[i], color='gray', alpha=0.2)
    plt.axhline(stop_loss, color='red', linestyle='--', label='Stop-Loss')
    plt.axhline(take_profit, color='green', linestyle='--', label='Take-Profit')
    plt.title(f'Simulated Stock Price Paths ({paths.shape[1]} Trading Days)')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def main():
    """
    Main function to run the Monte Carlo trade analysis.
    """
    # --- USER CONFIGURABLE PARAMETERS ---
    S0 = 100           # Current stock price
    mu = 0.10          # Expected annual return (10%)
    sigma = 0.20       # Annual volatility (20%)
    T_days = 30        # Holding period in trading days
    stop_loss = 90     # Stop-loss price
    take_profit = 120  # Take-profit price
    capital = 10000    # Total capital in dollars
    risk_per_trade = 0.02  # Risk 2% of capital per trade
    n_paths = 10000    # Number of Monte Carlo simulations

    # --- SIMULATION SETUP ---
    T = T_days / 252
    dt = 1 / 252

    stop_prob, target_prob, expected_return, paths = run_trade_simulation(
        S0, mu, sigma, T, dt, n_paths, stop_loss, take_profit
    )

    position_size = calculate_position_size(
        capital, risk_per_trade, stop_loss, S0
    )

    print("\n📊 Monte Carlo Simulation Results:")
    print(f"Stop-Loss Hit Probability: {stop_prob:.2%}")
    print(f"Take-Profit Hit Probability: {target_prob:.2%}")
    print(f"Expected Profit per Share: ${expected_return:.2f}")
    print(f"Suggested Position Size: {position_size:.2f} shares")
    print(f"Estimated Expected Profit: ${expected_return * position_size:.2f}")

    plot_simulation(paths, stop_loss, take_profit)


if __name__ == "__main__":
    main()
