import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import sys
import time

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the yield curve simulator
from yield_curve_simulation import YieldCurveSimulator

def main():
    st.title("Yield Curve Simulation Tool")
    
    st.sidebar.header("Simulation Parameters")
    
    # Simulation timeframe parameters
    st.sidebar.subheader("Timeframe")
    time_horizon = st.sidebar.slider("Time Horizon (Trading Days)", 50, 500, 250)
    num_steps = st.sidebar.slider("Number of Simulation Steps", 10, 100, 50)
    
    # Volatility parameters
    st.sidebar.subheader("Volatility Parameters")
    vol_short = st.sidebar.slider("Short-term Rate Volatility", 0.01, 0.50, 0.15, 0.01)
    vol_medium = st.sidebar.slider("Medium-term Rate Volatility", 0.01, 0.30, 0.10, 0.01)
    vol_long = st.sidebar.slider("Long-term Rate Volatility", 0.01, 0.20, 0.08, 0.01)
    vol_inflation = st.sidebar.slider("Inflation Volatility", 0.01, 0.15, 0.05, 0.01)
    
    # Drift parameters
    st.sidebar.subheader("Drift Parameters")
    drift_short = st.sidebar.slider("Short-term Rate Drift", -0.05, 0.05, 0.01, 0.005)
    drift_medium = st.sidebar.slider("Medium-term Rate Drift", -0.05, 0.05, 0.005, 0.005)
    drift_long = st.sidebar.slider("Long-term Rate Drift", -0.05, 0.05, 0.002, 0.005)
    drift_inflation = st.sidebar.slider("Inflation Drift", -0.05, 0.05, 0.02, 0.005)
    
    # Correlation matrix inputs
    st.sidebar.subheader("Correlation Parameters")
    corr_short_medium = st.sidebar.slider("Short-term/Medium-term Correlation", -1.0, 1.0, 0.8, 0.1)
    corr_short_long = st.sidebar.slider("Short-term/Long-term Correlation", -1.0, 1.0, 0.6, 0.1)
    corr_short_inflation = st.sidebar.slider("Short-term/Inflation Correlation", -1.0, 1.0, 0.5, 0.1)
    corr_medium_long = st.sidebar.slider("Medium-term/Long-term Correlation", -1.0, 1.0, 0.8, 0.1)
    corr_medium_inflation = st.sidebar.slider("Medium-term/Inflation Correlation", -1.0, 1.0, 0.7, 0.1)
    corr_long_inflation = st.sidebar.slider("Long-term/Inflation Correlation", -1.0, 1.0, 0.9, 0.1)
    
    # Initial rates
    st.sidebar.subheader("Initial Rate Parameters")
    initial_short = st.sidebar.slider("Initial Short-term Rate (%)", 0.0, 10.0, 3.0, 0.25) / 100
    initial_medium = st.sidebar.slider("Initial Medium-term Rate (%)", 0.0, 10.0, 4.0, 0.25) / 100
    initial_long = st.sidebar.slider("Initial Long-term Rate (%)", 0.0, 10.0, 5.0, 0.25) / 100
    initial_inflation = st.sidebar.slider("Initial Inflation Rate (%)", 0.0, 10.0, 2.0, 0.25) / 100
    
    # Add auto-update controls
    st.sidebar.subheader("Auto Update Settings")
    auto_update = st.sidebar.checkbox("Enable Auto Update", value=False)
    update_interval = st.sidebar.number_input("Update Interval (seconds)", min_value=1, value=5)
    steps_per_update = st.sidebar.number_input("Steps per Update", min_value=1, value=10, max_value=50)
    
    # Initialize session state if not exists
    if 'simulator' not in st.session_state:
        st.session_state.simulator = None
        st.session_state.current_step = 0
    
    # Build the correlation matrix from inputs
    correlation_matrix = np.array([
        [1.0, corr_short_medium, corr_short_long, corr_short_inflation],
        [corr_short_medium, 1.0, corr_medium_long, corr_medium_inflation],
        [corr_short_long, corr_medium_long, 1.0, corr_long_inflation],
        [corr_short_inflation, corr_medium_inflation, corr_long_inflation, 1.0]
    ])
    
    # Check if correlation matrix is positive semi-definite
    eigenvalues = np.linalg.eigvals(correlation_matrix)
    if not np.all(eigenvalues >= -1e-10):  # Allow for small numerical errors
        st.error("Warning: The correlation matrix is not positive semi-definite. Please adjust correlation parameters.")
        return
    
    # Run button
    run_simulation = st.sidebar.button("Run Simulation")
    stop_simulation = st.sidebar.button("Stop Simulation")
    
    if run_simulation or (auto_update and not stop_simulation):
        if run_simulation:
            # Reset simulation
            simulator = YieldCurveSimulator(time_horizon=time_horizon, num_steps=num_steps)
            simulator.volatilities = {
                'short_term': vol_short,
                'medium_term': vol_medium,
                'long_term': vol_long,
                'inflation': vol_inflation
            }
            simulator.drifts = {
                'short_term': drift_short,
                'medium_term': drift_medium,
                'long_term': drift_long,
                'inflation': drift_inflation
            }
            simulator.correlation_matrix = correlation_matrix
            st.session_state.simulator = simulator
            st.session_state.current_step = 0
        
        # Create placeholder for visualizations
        plot_container = st.empty()
        progress_bar = st.progress(0)
        
        while st.session_state.current_step < num_steps and auto_update and not stop_simulation:
            with plot_container.container():
                # Calculate next batch of steps
                end_step = min(st.session_state.current_step + steps_per_update, num_steps)
                st.session_state.simulator.simulate_steps(st.session_state.current_step, end_step)
                
                # Get current curves and display
                yield_curves, zero_rates_df = st.session_state.simulator.get_current_curves()
                
                # Display tabs
                tab1, tab2, tab3 = st.tabs(["Yield Curve Snapshots", "Rate Evolution", "Data Table"])
                
                with tab1:
                    st.subheader("Yield Curve Snapshots")
                    selected_steps = [0, end_step//2, end_step-1]
                    fig = st.session_state.simulator.plot_yield_curves(zero_rates_df, selected_steps)
                    st.pyplot(fig)
                
                with tab2:
                    st.subheader("Rate Evolution")
                    available_maturities = [0.5, 2, 5, 10]  # Example maturities
                    fig = st.session_state.simulator.plot_rate_evolution(zero_rates_df, available_maturities)
                    st.pyplot(fig)
                
                with tab3:
                    st.subheader("Zero Rates Data")
                    st.dataframe(zero_rates_df)
                
                # Update progress and step
                st.session_state.current_step = end_step
                progress_bar.progress(st.session_state.current_step / num_steps)
                
                time.sleep(update_interval)
        
        if not auto_update or st.session_state.current_step >= num_steps:
            # Final update for non-auto or completed simulation
            yield_curves, zero_rates_df = st.session_state.simulator.get_current_curves()
            # Display final state
            # ... existing visualization code ...
    
    else:
        st.info("Adjust parameters on the sidebar and click 'Run Simulation' to start.")
        
        # Display example yield curve
        st.subheader("Example Yield Curve")
        
        # Create a simple example curve
        maturities = [0.25, 0.5, 1, 2, 5, 10]
        rates = [initial_short, 
                initial_short * 1.1, 
                initial_medium, 
                initial_medium * 1.05, 
                initial_long * 0.95, 
                initial_long]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(maturities, rates, marker='o')
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Zero Rate')
        ax.set_title('Example Yield Curve')
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Information about the app
        st.markdown("""
        ## About This Tool
        
        This application simulates the evolution of yield curves over time using geometric Brownian motion with correlated parameters.
        
        ### Key Features:
        - Simulate yield curves based on financial instruments (bills and bonds)
        - Model correlation between different parts of the curve
        - Visualize curve snapshots at different time points
        - Track the evolution of rates for specific maturities
        
        ### How to Use:
        1. Adjust parameters in the sidebar to control simulation behavior
        2. Click "Run Simulation" to generate yield curves
        3. Explore different visualizations in the tabs
        """)

if __name__ == "__main__":
    main()
