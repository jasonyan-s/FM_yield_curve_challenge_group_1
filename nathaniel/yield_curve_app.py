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
    
    if run_simulation:
        simulator = YieldCurveSimulator(time_horizon=time_horizon, num_steps=num_steps)
        
        # Update simulator parameters
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
        
        if auto_update:
            placeholder = st.empty()
            progress_bar = st.progress(0)
            
            current_step = 0
            while current_step < num_steps:
                with placeholder.container():
                    # Calculate next batch of steps
                    end_step = min(current_step + steps_per_update, num_steps)
                    simulator.simulate_steps(current_step, end_step)
                    
                    # Update visualizations
                    yield_curves, zero_rates_df = simulator.get_current_curves()
                    
                    tab1, tab2, tab3 = st.tabs(["Yield Curve Snapshots", "Rate Evolution", "Data Table"])
                    
                    with tab1:
                        st.subheader("Yield Curve Snapshots")
                        # Build options for time steps
                        step_options = list(range(0, num_steps, max(1, num_steps//10)))
                        if (num_steps - 1) not in step_options:
                            step_options.append(num_steps - 1)
                        step_options = sorted(set(step_options))
                        # Build safe defaults (only those in options)
                        default_steps = [0, num_steps//2, num_steps-1]
                        default_steps = [s for s in default_steps if s in step_options]
                        selected_steps = st.multiselect(
                            "Select time steps to display", 
                            options=step_options,
                            default=default_steps
                        )
                        
                        if selected_steps:
                            fig = simulator.plot_yield_curves(zero_rates_df, selected_steps)
                            st.pyplot(fig)
                        else:
                            st.info("Please select at least one time step to display.")
                    
                    with tab2:
                        st.subheader("Rate Evolution Over Time")
                        available_maturities = [float(col) for col in zero_rates_df.columns if col != 'time_step']
                        selected_maturities = st.multiselect(
                            "Select maturities to display (years)",
                            options=available_maturities,
                            default=[min(available_maturities), 
                                     available_maturities[len(available_maturities)//2], 
                                     max(available_maturities)]
                        )
                        
                        if selected_maturities:
                            fig = simulator.plot_rate_evolution(zero_rates_df, selected_maturities)
                            st.pyplot(fig)
                        else:
                            st.info("Please select at least one maturity to display.")
                    
                    with tab3:
                        st.subheader("Zero Rates Data")
                        
                        # Create a readable dataframe for display
                        display_df = zero_rates_df.copy()
                        # Convert rates to percentage for better readability
                        for col in display_df.columns:
                            if col != 'time_step':
                                display_df[col] = display_df[col] * 100
                        
                        # Rename columns to add "Year" suffix
                        display_df = display_df.rename(columns={col: f"{col} Year" if col != 'time_step' else col 
                                                  for col in display_df.columns})
                        
                        st.dataframe(display_df)
                        
                        # Allow user to download the data
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            label="Download data as CSV",
                            data=csv,
                            file_name="yield_curve_simulation.csv",
                            mime="text/csv",
                        )
                
                # Update progress
                current_step = end_step
                progress_bar.progress(current_step / num_steps)
                
                if current_step < num_steps:
                    time.sleep(update_interval)
        else:
            # Regular single-run simulation
            yield_curves, zero_rates_df = simulator.simulate_yield_curves()
            
            if not yield_curves:
                st.error("Simulation failed. Please try different parameters.")
                return
                
            # Display simulation results
            st.success(f"Successfully simulated {num_steps} yield curves!")
            
            # Tabs for different visualizations
            tab1, tab2, tab3 = st.tabs(["Yield Curve Snapshots", "Rate Evolution", "Data Table"])
            
            with tab1:
                st.subheader("Yield Curve Snapshots")
                # Build options for time steps
                step_options = list(range(0, num_steps, max(1, num_steps//10)))
                if (num_steps - 1) not in step_options:
                    step_options.append(num_steps - 1)
                step_options = sorted(set(step_options))
                # Build safe defaults (only those in options)
                default_steps = [0, num_steps//2, num_steps-1]
                default_steps = [s for s in default_steps if s in step_options]
                selected_steps = st.multiselect(
                    "Select time steps to display", 
                    options=step_options,
                    default=default_steps
                )
                
                if selected_steps:
                    fig = simulator.plot_yield_curves(zero_rates_df, selected_steps)
                    st.pyplot(fig)
                else:
                    st.info("Please select at least one time step to display.")
            
            with tab2:
                st.subheader("Rate Evolution Over Time")
                available_maturities = [float(col) for col in zero_rates_df.columns if col != 'time_step']
                selected_maturities = st.multiselect(
                    "Select maturities to display (years)",
                    options=available_maturities,
                    default=[min(available_maturities), 
                             available_maturities[len(available_maturities)//2], 
                             max(available_maturities)]
                )
                
                if selected_maturities:
                    fig = simulator.plot_rate_evolution(zero_rates_df, selected_maturities)
                    st.pyplot(fig)
                else:
                    st.info("Please select at least one maturity to display.")
            
            with tab3:
                st.subheader("Zero Rates Data")
                
                # Create a readable dataframe for display
                display_df = zero_rates_df.copy()
                # Convert rates to percentage for better readability
                for col in display_df.columns:
                    if col != 'time_step':
                        display_df[col] = display_df[col] * 100
                
                # Rename columns to add "Year" suffix
                display_df = display_df.rename(columns={col: f"{col} Year" if col != 'time_step' else col 
                                              for col in display_df.columns})
                
                st.dataframe(display_df)
                
                # Allow user to download the data
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="yield_curve_simulation.csv",
                    mime="text/csv",
                )
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
