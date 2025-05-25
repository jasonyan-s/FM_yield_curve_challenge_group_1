import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
import streamlit as st

# Import the existing modules
import curve_classes_and_functions as curves
import instrument_classes as instruments

class YieldCurveSimulator:
    """
    A class to simulate the movement of yield curves based on changes in
    instrument parameters using geometric Brownian motion.
    """

    def __init__(self, time_horizon=250, num_steps=50, dt=1/250,
                 initial_short_rate=0.03, initial_medium_rate=0.04,
                 initial_long_rate=0.05, initial_inflation_rate=0.02):
        """
        Initialize the simulator with parameters for the simulation.

        Parameters:
        -----------
        time_horizon : int
            Number of trading days to simulate
        num_steps : int
            Number of time steps in the simulation
        dt : float
            Time increment for each step (default: 1/250 for daily steps in a trading year)
        initial_short_rate : float
            Initial short-term rate
        initial_medium_rate : float
            Initial medium-term rate
        initial_long_rate : float
            Initial long-term rate
        initial_inflation_rate : float
            Initial inflation rate
        """
        self.time_horizon = time_horizon
        self.num_steps = num_steps
        self.dt = dt

        # Default correlation parameters (can be overridden by Streamlit UI)
        self.correlation_matrix = np.array([
            [1.0, 0.8, 0.6, 0.5],  # Short-term rates
            [0.8, 1.0, 0.8, 0.7],  # Medium-term rates
            [0.6, 0.8, 1.0, 0.9],  # Long-term rates
            [0.5, 0.7, 0.9, 1.0]   # Inflation
        ])

        # Default volatility parameters (can be overridden by Streamlit UI)
        self.volatilities = {
            'short_term': 0.15,
            'medium_term': 0.10,
            'long_term': 0.08,
            'inflation': 0.05
        }

        # Default drift parameters (can be overridden by Streamlit UI)
        self.drifts = {
            'short_term': 0.01,
            'medium_term': 0.005,
            'long_term': 0.002,
            'inflation': 0.02
        }

        # Initial rates
        self.initial_rates = {
            'short_term': initial_short_rate,
            'medium_term': initial_medium_rate,
            'long_term': initial_long_rate,
            'inflation': initial_inflation_rate
        }

        # Current state for step-by-step simulation
        self.current_rates = None
        self.rate_paths = None
        self.yield_curves = []
        self.zero_rates_history = []

    def generate_correlated_gbm_paths(self):
        """
        Generate correlated geometric Brownian motion paths for all parameters.

        Returns:
        --------
        dict: Dictionary of simulated paths for each parameter
        """
        # Number of variables
        n_vars = len(self.correlation_matrix)

        # Cholesky decomposition for correlation
        L = np.linalg.cholesky(self.correlation_matrix)

        # Initialize paths with initial rates from constructor/Streamlit
        paths = {
            'short_term': np.zeros(self.num_steps),
            'medium_term': np.zeros(self.num_steps),
            'long_term': np.zeros(self.num_steps),
            'inflation': np.zeros(self.num_steps)
        }

        # Set initial values from the simulator's initial_rates
        paths['short_term'][0] = self.initial_rates['short_term']
        paths['medium_term'][0] = self.initial_rates['medium_term']
        paths['long_term'][0] = self.initial_rates['long_term']
        paths['inflation'][0] = self.initial_rates['inflation']

        # Create the correlated random variables
        for t in range(1, self.num_steps):
            # Generate uncorrelated normal random variables
            z = np.random.normal(0, 1, n_vars)

            # Apply correlation using Cholesky decomposition
            correlated_z = np.dot(L, z)

            # Update each path with geometric Brownian motion
            # Ensure rates remain positive
            paths['short_term'][t] = max(1e-6, paths['short_term'][t-1] * np.exp(
                (self.drifts['short_term'] - 0.5 * self.volatilities['short_term']**2) * self.dt +
                self.volatilities['short_term'] * np.sqrt(self.dt) * correlated_z[0]
            ))

            paths['medium_term'][t] = max(1e-6, paths['medium_term'][t-1] * np.exp(
                (self.drifts['medium_term'] - 0.5 * self.volatilities['medium_term']**2) * self.dt +
                self.volatilities['medium_term'] * np.sqrt(self.dt) * correlated_z[1]
            ))

            paths['long_term'][t] = max(1e-6, paths['long_term'][t-1] * np.exp(
                (self.drifts['long_term'] - 0.5 * self.volatilities['long_term']**2) * self.dt +
                self.volatilities['long_term'] * np.sqrt(self.dt) * correlated_z[2]
            ))

            paths['inflation'][t] = max(1e-6, paths['inflation'][t-1] * np.exp(
                (self.drifts['inflation'] - 0.5 * self.volatilities['inflation']**2) * self.dt +
                self.volatilities['inflation'] * np.sqrt(self.dt) * correlated_z[3]
            ))

        return paths

    def create_instruments_from_rates(self, rates):
        """
        Create a portfolio of financial instruments based on simulated rates.
        We will define 'market-implied' YTMs for these instruments
        based on the simulated rates for different maturities.
        Then, we calculate their prices.
        
        Parameters:
        -----------
        rates : dict
            Dictionary of simulated rates for different 'tenors' (short, medium, long, inflation).
            These represent a proxy for where the market believes rates *should* be.

        Returns:
        --------
        instruments.Portfolio: Portfolio containing bills and bonds with calculated prices.
        """
        portfolio = instruments.Portfolio()

        # Helper function to get a representative rate for a maturity
        def get_rate_for_maturity(maturity):
            # Create smoother rate transitions
            short_weight = max(0, min(1, (0.5 - maturity) / 0.5))
            medium_weight = max(0, min(1, (5 - maturity) / 4.5)) if maturity > 0.5 else 0
            long_weight = max(0, min(1, (maturity - 1) / 9)) if maturity > 1 else 0
            
            total_weight = short_weight + medium_weight + long_weight
            if total_weight > 0:
                rate = (rates['short_term'] * short_weight + 
                       rates['medium_term'] * medium_weight + 
                       rates['long_term'] * long_weight) / total_weight
            else:
                rate = rates['medium_term']  # fallback
            
            return rate

        # Create and initialize bills
        bill1 = instruments.Bank_bill(face_value=100, maturity=0.25)
        bill1.set_ytm(get_rate_for_maturity(0.25))
        bill1.set_cash_flows()
        bill1.price = bill1.get_price()
        portfolio.add_bank_bill(bill1)

        bill2 = instruments.Bank_bill(face_value=100, maturity=0.5)
        bill2.set_ytm(get_rate_for_maturity(0.5))
        bill2.set_cash_flows()
        bill2.price = bill2.get_price()
        portfolio.add_bank_bill(bill2)

        # Create and initialize bonds with appropriate market rates
        bond_maturities = [1, 2, 3, 5, 7, 10]
        for maturity in bond_maturities:
            market_rate = get_rate_for_maturity(maturity)
            bond = instruments.Bond(
                face_value=100,
                maturity=maturity,
                coupon=market_rate,  # Set coupon to market rate
                frequency=2
            )
            bond.set_ytm(market_rate)
            bond.set_cash_flows()
            bond.price = bond.get_price()  # Calculate price based on YTM
            portfolio.add_bond(bond)

        # Create inflation-linked bond
        inflation_bond = instruments.InflationLinkedBond(
            face_value=100,
            maturity=7,
            coupon=max(0.015, get_rate_for_maturity(7) - rates['inflation']),  # Real yield
            frequency=2,
            inflation_rate=rates['inflation']
        )
        inflation_bond.set_ytm(get_rate_for_maturity(7))
        inflation_bond.set_cash_flows()
        inflation_bond.price = inflation_bond.get_price()
        portfolio.add_inflation_linked_bond(inflation_bond)

        # Set all cash flows for the portfolio
        portfolio.set_cash_flows()
        return portfolio

    def construct_yield_curve(self, portfolio):
        """
        Construct a yield curve from a portfolio of instruments.

        Parameters:
        -----------
        portfolio : instruments.Portfolio
            Portfolio of financial instruments

        Returns:
        --------
        curves.YieldCurve: Bootstrapped yield curve
        """
        yc = curves.YieldCurve()
        yc.set_constituent_portfolio(portfolio)

        try:
            yc.bootstrap()
            return yc
        except ZeroDivisionError as zde:
            st.error(f"Divide by zero error during bootstrapping: {zde}. This can happen if instrument prices or cash flows are zero.")
            return None
        except ValueError as ve:
            st.error(f"Value error during bootstrapping: {ve}. This might indicate invalid cash flows or prices.")
            return None
        except Exception as e:
            st.error(f"Unexpected error during bootstrapping: {e}")
            return None

    def simulate_steps(self, start_step, end_step):
        """
        Simulate yield curves for a specific range of steps.
        
        Parameters:
        -----------
        start_step : int
            Starting step index
        end_step : int
            Ending step index (exclusive)
        """
        if self.rate_paths is None:
            # Initialize paths if first time
            self.rate_paths = self.generate_correlated_gbm_paths()
        
        for t in range(start_step, end_step):
            # Extract rates for current step
            current_rates = {
                'short_term': self.rate_paths['short_term'][t],
                'medium_term': self.rate_paths['medium_term'][t],
                'long_term': self.rate_paths['long_term'][t],
                'inflation': self.rate_paths['inflation'][t]
            }
            
            # Create instruments and construct yield curve
            portfolio = self.create_instruments_from_rates(current_rates)
            yc = self.construct_yield_curve(portfolio)
            
            if yc is not None:
                self.yield_curves.append(yc)
                # Extract and store zero rates
                maturities_to_extract = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
                zero_rates_at_step = {'time_step': t}
                for m in maturities_to_extract:
                    try:
                        rate = yc.get_zero_rate(m)
                        zero_rates_at_step[str(m)] = max(0.0001, min(rate, 2.0))
                    except ValueError:
                        zero_rates_at_step[str(m)] = np.nan
                self.zero_rates_history.append(zero_rates_at_step)
    
    def get_current_curves(self):
        """
        Get the current state of yield curves and rates.
        
        Returns:
        --------
        tuple: (yield_curves, zero_rates_df)
        """
        zero_rates_df = pd.DataFrame(self.zero_rates_history)
        return self.yield_curves, zero_rates_df
    
    def simulate_yield_curves(self):
        """Run complete simulation."""
        self.yield_curves = []
        self.zero_rates_history = []
        self.simulate_steps(0, self.num_steps)
        return self.get_current_curves()

    def plot_yield_curves(self, zero_rates_df, selected_time_steps=None):
        """Plot initial and current yield curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get maturity columns and convert to floats for plotting
        maturities = [float(col) for col in zero_rates_df.columns if col != 'time_step']
        maturities.sort()
        
        # Plot only initial and current curves
        time_steps = [zero_rates_df['time_step'].min(), zero_rates_df['time_step'].max()]
        labels = ['Initial Curve', 'Current Curve']
        colors = ['blue', 'red']
        
        for step, label, color in zip(time_steps, labels, colors):
            curve_data = zero_rates_df[zero_rates_df['time_step'] == step]
            
            if not curve_data.empty:
                rates = [curve_data[str(m)].values[0] for m in maturities]
                ax.plot(maturities, rates, marker='o', label=label, color=color)
        
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Zero Rate')
        ax.set_title('Yield Curve Evolution')
        ax.legend()
        ax.grid(True)
        
        return fig

    def plot_rate_evolution(self, zero_rates_df, selected_maturities=None):
        """
        Plot the evolution of rates for selected maturities over time.

        Parameters:
        -----------
        zero_rates_df : pd.DataFrame
            DataFrame of zero rates for each maturity and time step
        selected_maturities : list, optional
            List of maturities to plot (default: 0.5, 2, and 10 years)

        Returns:
        --------
        matplotlib.figure.Figure: Rate evolution plot
        """
        if selected_maturities is None:
            selected_maturities = [0.5, 2, 10]

        fig, ax = plt.subplots(figsize=(10, 6))

        for maturity in selected_maturities:
            if str(maturity) in zero_rates_df.columns:
                ax.plot(
                    zero_rates_df['time_step'],
                    zero_rates_df[str(maturity)],
                    label=f'{maturity} Year'
                )

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Zero Rate')
        ax.set_title('Evolution of Rates Over Time')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(bottom=0) # Ensure y-axis starts from 0 for rates
        return fig

# This section is for local testing and will not be directly executed by Streamlit
# if __name__ == "__main__":
#     st.set_page_config(layout="wide") # Use wide layout for better visualization
#     st.title("Yield Curve Simulator Test")
#
#     # Example usage:
#     sim = YieldCurveSimulator(num_steps=100)
#     yield_curves, zero_rates_df = sim.simulate_yield_curves()
#
#     if not zero_rates_df.empty:
#         st.subheader("Simulated Zero Rates Data")
#         st.dataframe(zero_rates_df.head())
#
#         st.subheader("Yield Curve Snapshots (Example)")
#         fig_curves = sim.plot_yield_curves(zero_rates_df)
#         st.pyplot(fig_curves)
#
#         st.subheader("Rate Evolution (Example)")
#         fig_evolution = sim.plot_rate_evolution(zero_rates_df)
#         st.pyplot(fig_evolution)
#     else:
#         st.warning("No yield curve data generated. Check for simulation errors.")