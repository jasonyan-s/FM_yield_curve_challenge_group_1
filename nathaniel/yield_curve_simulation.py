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

        # Helper function to get a representative rate for a given maturity
        # This is a simplification; in reality, we'd use an existing curve to price.
        # Here, we're building the curve, so we use the simulated rates as anchors.
        def get_rate_for_maturity(maturity):
            if maturity <= 0.5:
                return rates['short_term']
            elif maturity <= 2:
                return rates['medium_term']
            else:
                return rates['long_term']

        # Define instruments with maturities and derive their market prices
        # based on the simulated rates.
        # Bank bills (zero-coupon)
        bill1 = instruments.Bank_bill(face_value=100, maturity=0.25)
        # Use the rate for its maturity to set its YTM for pricing
        bill1_ytm_for_pricing = get_rate_for_maturity(bill1.maturity)
        bill1.set_ytm(bill1_ytm_for_pricing)
        bill1.set_cash_flows()
        # The price is calculated based on its YTM
        bill1.price = bill1.get_price_from_ytm() # Store price attribute for bootstrap to use

        bill2 = instruments.Bank_bill(face_value=100, maturity=0.5)
        bill2_ytm_for_pricing = get_rate_for_maturity(bill2.maturity)
        bill2.set_ytm(bill2_ytm_for_pricing)
        bill2.set_cash_flows()
        bill2.price = bill2.get_price_from_ytm()

        # Bonds (coupon-bearing)
        # The coupon rate for bonds can be a fixed value or related to the simulated rates.
        # For simplicity, let's make coupon rates reflective of current market conditions.
        # We will use the simulated rates to set YTMs for pricing.

        bond1 = instruments.Bond(face_value=100, maturity=1, coupon=0.03, frequency=2)
        bond1_ytm_for_pricing = get_rate_for_maturity(bond1.maturity)
        bond1.set_ytm(bond1_ytm_for_pricing)
        bond1.set_cash_flows()
        bond1.price = bond1.get_price()

        bond2 = instruments.Bond(face_value=100, maturity=2, coupon=0.04, frequency=2)
        bond2_ytm_for_pricing = get_rate_for_maturity(bond2.maturity)
        bond2.set_ytm(bond2_ytm_for_pricing)
        bond2.set_cash_flows()
        bond2.price = bond2.get_price()

        bond3 = instruments.Bond(face_value=100, maturity=5, coupon=0.045, frequency=2)
        bond3_ytm_for_pricing = get_rate_for_maturity(bond3.maturity)
        bond3.set_ytm(bond3_ytm_for_pricing)
        bond3.set_cash_flows()
        bond3.price = bond3.get_price()

        bond4 = instruments.Bond(face_value=100, maturity=10, coupon=0.05, frequency=2)
        bond4_ytm_for_pricing = get_rate_for_maturity(bond4.maturity)
        bond4.set_ytm(bond4_ytm_for_pricing)
        bond4.set_cash_flows()
        bond4.price = bond4.get_price()

        # Inflation-linked bond example (using inflation rate for its effect)
        # For an inflation-linked bond, its cash flows are adjusted by inflation.
        # Its price would be discounted by real rates or a combination of nominal rates and inflation expectations.
        # For simplicity, we'll assign a real yield related to the long-term rate and
        # let its cash flows reflect inflation.
        inflation_linked_bond = instruments.InflationLinkedBond(
            face_value=100, maturity=7, coupon=0.015, frequency=2, inflation_rate=rates['inflation']
        )
        inflation_linked_bond.set_inflation_rate(rates['inflation'])
        # The YTM for pricing should represent a real yield, which we can tie to the long-term rate
        inflation_linked_bond_ytm_for_pricing = rates['long_term'] - rates['inflation']
        inflation_linked_bond.set_ytm(max(1e-6, inflation_linked_bond_ytm_for_pricing)) # Ensure positive YTM
        inflation_linked_bond.set_cash_flows()
        inflation_linked_bond.price = inflation_linked_bond.get_price()


        # Add all instruments to the portfolio
        portfolio.add_bank_bill(bill1)
        portfolio.add_bank_bill(bill2)
        portfolio.add_bond(bond1)
        portfolio.add_bond(bond2)
        portfolio.add_bond(bond3)
        portfolio.add_bond(bond4)
        portfolio.add_inflation_linked_bond(inflation_linked_bond)

        # It's crucial that instruments have a 'price' attribute for the bootstrap method.
        # We can dynamically add this attribute or modify instrument_classes to include it.
        # For now, we'll assume the get_price() method is correctly overridden/implemented
        # in each instrument class and returns a price that makes sense for bootstrapping.

        # The get_price() method of Instrument calls self.ytm if discount_rates is None.
        # We need to ensure each instrument *has* a price before bootstrapping,
        # and that price should be consistent with the simulated market conditions.
        # Let's ensure the get_price() method for bootstrap uses the specific price attribute we set.
        # To do this, we need to ensure the `bootstrap` method in `YieldCurve` can access this `price`.
        # The simplest way is to temporarily assign the calculated price to the instrument
        # so `bootstrap` can pick it up.

        for inst in portfolio.bills + portfolio.bonds + portfolio.inflation_linked_bonds:
            if not hasattr(inst, 'price') or inst.price is None:
                raise ValueError(f"Instrument {type(inst).__name__} does not have a price attribute.")
            if inst.price <= 0: # Ensure positive prices for bootstrapping
                st.warning(f"Instrument {type(inst).__name__} has non-positive price ({inst.price}), adjusting to 100 for bootstrap stability.")
                inst.price = 100.0 # Fallback to a default price

        # We must ensure that the `get_price()` method in `instrument_classes.py`
        # when called by `YieldCurve.bootstrap()` actually returns the *market price* we just set.
        # The current implementation of `get_price` in `Instrument` bases it on YTM.
        # So, we need to ensure the YTM is consistent with the simulated market rates.
        # The problem is that `bootstrap` is trying to infer the YTM from price, not the other way around.
        # Let's modify the Instrument.get_price to prioritize a set 'market_price' if it exists.
        # This requires a small change to instrument_classes.py, but for now, the approach above
        # (setting inst.price directly and ensuring bootstrap can use it) is the target.

        # The `bootstrap` method in `YieldCurve` currently calls `inst.get_price()`
        # and if discount_rates is None, it uses the instrument's internal `ytm`.
        # The issue is that we are setting `ytm` to what we *think* the market rate is for pricing,
        # and then `bootstrap` tries to infer the zero rates *from* these prices.
        # This implies that the `ytm` we set is indeed the "market" YTM of that instrument.

        # The critical part is that when `bootstrap` calls `inst.get_price()`, it needs to
        # correctly reflect the market price of the bond that is consistent with the yields
        # it is trying to derive.

        # A better approach: The `bootstrap` method should directly use the prices of the instruments.
        # Currently, it calls `inst.get_price()` without passing `discount_rates`.
        # This means `inst.get_price()` will use its own internal `self.ytm`.
        # We need to ensure `self.ytm` for each instrument is set to a value
        # that, when used to calculate `get_price()`, results in a price that accurately
        # reflects a point on the "market curve" implied by `current_rates`.

        # So, we are setting the YTMs of the instruments to reflect the simulated market rates.
        # Then, their `get_price()` methods will correctly calculate their "market prices"
        # based on these YTMs. The bootstrapping process then takes these "market prices"
        # and their cash flows to build the zero curve.

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

    def simulate_yield_curves(self):
        """
        Simulate a series of yield curves based on parameter paths.

        Returns:
        --------
        list: List of yield curves at each time step
        pd.DataFrame: DataFrame of zero rates for each maturity at each time step
        """
        # Generate rate paths
        rate_paths = self.generate_correlated_gbm_paths()

        # Store yield curves and zero rates for each time step
        yield_curves = []
        zero_rates_data = []

        # Fixed maturities to extract for each curve
        # These should align with typical points on a yield curve and instrument maturities.
        maturities_to_extract = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]

        for t in range(self.num_steps):
            # Extract rates at this time step
            current_rates = {
                'short_term': rate_paths['short_term'][t],
                'medium_term': rate_paths['medium_term'][t],
                'long_term': rate_paths['long_term'][t],
                'inflation': rate_paths['inflation'][t]
            }

            # Create instruments and construct yield curve
            portfolio = self.create_instruments_from_rates(current_rates)
            yc = self.construct_yield_curve(portfolio)

            if yc is not None:
                yield_curves.append(yc)

                # Extract zero rates for specific maturities
                zero_rates_at_step = {'time_step': t}
                for m in maturities_to_extract:
                    try:
                        rate = yc.get_zero_rate(m)
                        # Ensure rates are not negative or excessively large
                        zero_rates_at_step[str(m)] = max(0.0001, min(rate, 2.0)) # Clamp between 0.01% and 200%
                    except ValueError:
                        zero_rates_at_step[str(m)] = np.nan # Handle cases where rate might be undefined
                zero_rates_data.append(zero_rates_at_step)

        # Convert zero rates to DataFrame
        zero_rates_df = pd.DataFrame(zero_rates_data)

        return yield_curves, zero_rates_df

    def plot_yield_curves(self, zero_rates_df, selected_time_steps=None):
        """Plot yield curves for selected time steps."""
        if selected_time_steps is None:
            time_steps = zero_rates_df['time_step'].unique()
            selected_time_steps = [time_steps[0], time_steps[len(time_steps)//2], time_steps[-1]]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get maturity columns and convert to floats for plotting
        maturities = [float(col) for col in zero_rates_df.columns if col != 'time_step']
        maturities.sort()  # Ensure maturities are in ascending order
        
        for step in selected_time_steps:
            curve_data = zero_rates_df[zero_rates_df['time_step'] == step]
            
            if not curve_data.empty:
                # Access DataFrame columns using original column names
                rates = []
                for m in maturities:
                    # Find the exact column name that matches this maturity
                    col_name = next(col for col in zero_rates_df.columns 
                                  if col != 'time_step' and abs(float(col) - m) < 1e-10)
                    rates.append(curve_data[col_name].values[0])
                
                ax.plot(maturities, rates, marker='o', label=f'Time Step {step}')
        
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Zero Rate')
        ax.set_title('Simulated Yield Curves Over Time')
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