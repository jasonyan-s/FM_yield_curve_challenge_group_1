import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from simulation.gbm_simulator import GBMSimulator
from market.interest_rate_market import InterestRateMarket
from visualization.curve_visualizer import CurveVisualizer
from arbitrage.detector import ArbitrageDetector
from config.simulation_config import (
    SIMULATION_STEPS, TIME_STEP, BANK_BILLS, BONDS,
    GBM_PARAMETERS, CORRELATION_MATRIX, VISUALIZATION
)

# Set page configuration
st.set_page_config(
    page_title="Interest Rate Market Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("Interest Rate Market Simulation")
st.write("""
This application simulates an interest rate market with various instruments 
(bank bills and bonds) and visualizes the evolution of the yield curve over time.
The simulation uses Geometric Brownian Motion to model rate movements and
detects potential arbitrage opportunities.
""")

# Sidebar for simulation controls
st.sidebar.header("Simulation Controls")

# Simulation parameters
steps = st.sidebar.slider("Number of simulation steps", 10, 200, SIMULATION_STEPS)
update_speed = st.sidebar.slider(
    "Animation speed (ms)", 
    1000, 10000, VISUALIZATION['animation_interval'], step=1000
)

# Initialize session state
if 'simulation_state' not in st.session_state:
    st.session_state.simulation_state = 'initial'
    st.session_state.current_step = 0
    st.session_state.curve_history = []

# GBM parameters adjustment
st.sidebar.header("GBM Parameters")

# Allow adjustment of GBM parameters for a specific maturity
selected_maturity = st.sidebar.selectbox(
    "Select maturity to adjust GBM parameters",
    list(GBM_PARAMETERS.keys())
)

if selected_maturity:
    default_mu, default_sigma = GBM_PARAMETERS[selected_maturity]
    mu_adj = st.sidebar.slider(f"Drift (μ) for {selected_maturity}y", -0.01, 0.01, float(default_mu), 0.001)
    sigma_adj = st.sidebar.slider(f"Volatility (σ) for {selected_maturity}y", 0.001, 0.1, float(default_sigma), 0.001)
    GBM_PARAMETERS[selected_maturity] = (mu_adj, sigma_adj)

# Initialize the market
@st.cache_resource
def initialize_market():
    market = InterestRateMarket()
    
    # Add bank bills
    for maturity, ytm, face_value in BANK_BILLS:
        market.add_bank_bill(maturity, ytm, face_value)
    
    # Add bonds
    for maturity, ytm, coupon, frequency, face_value in BONDS:
        market.add_bond(maturity, ytm, coupon, frequency, face_value)
    
    # Build the yield curve
    market.build_yield_curve()
    
    return market

market = initialize_market()

# Initialize GBM simulators for each maturity
@st.cache_resource
def initialize_simulators():
    simulators = {}
    for maturity in GBM_PARAMETERS:
        mu, sigma = GBM_PARAMETERS[maturity]
        # Find initial rate for this maturity
        initial_rate = None
        for bill in market.bank_bills:
            if bill.get_maturity() == maturity:
                initial_rate = bill.get_ytm()
                break
        for bond in market.bonds:
            if bond.get_maturity() == maturity:
                initial_rate = bond.get_ytm()
                break
        
        if initial_rate is None:
            # If no exact match, use the first available rate
            if market.bank_bills:
                initial_rate = market.bank_bills[0].get_ytm()
            elif market.bonds:
                initial_rate = market.bonds[0].get_ytm()
            else:
                initial_rate = 0.05  # Default
        
        simulators[maturity] = GBMSimulator(initial_rate, mu, sigma, TIME_STEP)
    
    return simulators

simulators = initialize_simulators()

# Initialize the arbitrage detector
arbitrage_detector = ArbitrageDetector(threshold=0.002)

# Initialize the curve visualizer
visualizer = CurveVisualizer(figsize=VISUALIZATION['figsize'])

# Function to reset simulation
def reset_simulation():
    st.session_state.simulation_state = 'initial'
    st.session_state.current_step = 0
    visualizer.curve_history = []
    
    # Reset simulators
    for maturity, simulator in simulators.items():
        mu, sigma = GBM_PARAMETERS[maturity]
        simulator.mu = mu
        simulator.sigma = sigma
        simulator.reset()

# Button to start/reset simulation
if st.sidebar.button("Reset Simulation"):
    reset_simulation()
    st.rerun()

# Function to run one step of the simulation
def run_simulation_step():
    """Run one step of the simulation with error handling."""
    try:
        # Generate correlated random changes
        rate_changes = {}
        
        # For each maturity, get the next simulated rate
        for maturity, simulator in simulators.items():
            try:
                if len(simulator.history) >= 2:
                    rate_changes[maturity] = simulator.next_step() / simulator.history[-2]
                else:
                    # If we don't have enough history, just use a small random change
                    rate_changes[maturity] = 1.0 + (np.random.random() * 0.01 - 0.005)
            except Exception as e:
                print(f"Error updating rate for maturity {maturity}: {e}")
                # Apply a default small change if simulation fails
                rate_changes[maturity] = 1.0
        
        # Update instrument rates
        market.update_instrument_rates(rate_changes)
        
        # Rebuild the yield curve
        market.build_yield_curve()
        
        # Update step counter
        st.session_state.current_step += 1
        
        # Check if simulation should end
        if st.session_state.current_step >= steps:
            st.session_state.simulation_state = 'finished'
        
        return rate_changes
        
    except Exception as e:
        st.error(f"Error in simulation step: {e}")
        # Provide a default return value so the app continues
        return {maturity: 1.0 for maturity in GBM_PARAMETERS.keys()}

# Main area layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Yield Curve Visualization")
    
    # Create placeholder for the yield curve plot
    plot_placeholder = st.empty()
    
    # Current state info
    step_info = st.empty()

with col2:
    st.subheader("Arbitrage Opportunities")
    
    # Create placeholder for arbitrage information
    arbitrage_placeholder = st.empty()
    
    st.subheader("Market Instruments")
    instrument_placeholder = st.empty()

# Add simulation mode selection
simulation_mode = st.sidebar.radio(
    "Simulation Mode",
    ["Manual Step", "Auto-Run", "Pre-compute & Navigate"],
    index=0
)

# Simulation Logic
if simulation_mode == "Pre-compute & Navigate":
    # Pre-compute the entire simulation
    if st.sidebar.button("Pre-compute All Steps") or ('precomputed_data' not in st.session_state and st.session_state.simulation_state == 'initial'):
        with st.spinner("Pre-computing simulation steps..."):
            # Reset simulation
            reset_simulation()
            st.session_state.precomputed_data = []
            
            # Run all steps
            for _ in range(steps):
                run_simulation_step()
                
                # Get current curve
                maturities, discount_factors = market.yield_curve.get_zero_curve()
                
                if len(maturities) > 0 and len(discount_factors) > 0:
                    # Convert to zero rates
                    zero_rates = []
                    for i, mat in enumerate(maturities):
                        if mat > 0:
                            zero_rates.append(-np.log(discount_factors[i])/mat)
                        else:
                            zero_rates.append(0)
                    
                    # Store the data
                    step_data = {
                        'step': st.session_state.current_step,
                        'maturities': maturities.copy(),
                        'zero_rates': zero_rates.copy(),
                        'curve_arbitrage': arbitrage_detector.detect_curve_arbitrage(market.yield_curve),
                        'cross_arbitrage': arbitrage_detector.check_cross_instrument_arbitrage(market),
                        'bank_bills': [(b.get_maturity(), b.get_ytm(), b.get_price()) for b in market.bank_bills],
                        'bonds': [(b.get_maturity(), b.get_coupon_rate(), b.get_ytm(), b.get_price()) for b in market.bonds]
                    }
                    st.session_state.precomputed_data.append(step_data)
                    
                    # Add to visualizer
                    visualizer.add_curve(maturities, zero_rates)
            
            # Mark as precomputed
            st.session_state.simulation_state = 'precomputed'
    
    # Show a slider to navigate precomputed data
    if 'precomputed_data' in st.session_state and len(st.session_state.precomputed_data) > 0:
        selected_step = st.sidebar.slider(
            "Select Step", 0, len(st.session_state.precomputed_data) - 1, 0
        )
        
        # Display the selected step data
        data = st.session_state.precomputed_data[selected_step]
        
        # Plot the curve
        fig, ax = plt.subplots(figsize=VISUALIZATION['figsize'])
        ax.plot(data['maturities'], data['zero_rates'], 'o-', linewidth=2, markersize=8)
        ax.set_title(f"Yield Curve - Step {selected_step+1}/{steps}")
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Zero Rate')
        ax.grid(True)
        plot_placeholder.pyplot(fig)
        
        # Show step info
        step_info.write(f"### Step {selected_step+1}/{steps}")
        
        # Show arbitrage info
        with arbitrage_placeholder.container():
            curve_arbitrage = data['curve_arbitrage']
            cross_arbitrage = data['cross_arbitrage']
            
            if curve_arbitrage:
                st.warning(f"Found {len(curve_arbitrage)} curve arbitrage opportunities!")
                for opp in curve_arbitrage:
                    st.write(f"""
                    **Curve Arbitrage**: Between {opp['maturity_short']}y and {opp['maturity_long']}y maturities  
                    Potential profit: {opp['potential']:.2%}
                    """)
            
            if cross_arbitrage:
                st.warning(f"Found {len(cross_arbitrage)} cross-instrument arbitrage opportunities!")
                for opp in cross_arbitrage:
                    st.write(f"""
                    **Instrument Arbitrage**: Between {opp['instrument1']} and {opp['instrument2']}  
                    YTMs: {opp['ytm1']:.2%} vs {opp['ytm2']:.2%}  
                    Potential profit: {opp['potential']:.2%}
                    """)
            
            if not curve_arbitrage and not cross_arbitrage:
                st.success("No arbitrage opportunities detected in this step.")
        
        # Show instruments
        with instrument_placeholder.container():
            # Bank bills
            if data['bank_bills']:
                st.write("**Bank Bills:**")
                bill_data = {
                    "Maturity": [b[0] for b in data['bank_bills']],
                    "YTM": [f"{b[1]:.2%}" for b in data['bank_bills']],
                    "Price": [f"${b[2]:.2f}" for b in data['bank_bills']]
                }
                st.dataframe(pd.DataFrame(bill_data))
            
            # Bonds
            if data['bonds']:
                st.write("**Bonds:**")
                bond_data = {
                    "Maturity": [b[0] for b in data['bonds']],
                    "Coupon": [f"{b[1]:.2%}" for b in data['bonds']],
                    "YTM": [f"{b[2]:.2%}" for b in data['bonds']],
                    "Price": [f"${b[3]:.2f}" for b in data['bonds']]
                }
                st.dataframe(pd.DataFrame(bond_data))
    else:
        st.warning("No precomputed data available. Click 'Pre-compute All Steps' to generate the simulation data.")

elif simulation_mode == "Manual Step":
    # Manual stepping mode
    if st.sidebar.button("Run Next Step"):
        if st.session_state.current_step < steps:
            # Run one step
            run_simulation_step()
            st.rerun()
    
    # Show the current state
    if st.session_state.current_step > 0:
        # Get the current yield curve
        maturities, discount_factors = market.yield_curve.get_zero_curve()
        
        if len(maturities) > 0 and len(discount_factors) > 0:
            # Convert discount factors to zero rates for display
            zero_rates = []
            for i, mat in enumerate(maturities):
                if mat > 0:
                    zero_rates.append(-np.log(discount_factors[i])/mat)
                else:
                    zero_rates.append(0)
            
            # Add the curve to the visualizer history
            visualizer.add_curve(maturities, zero_rates)
            
            # Create the current curve plot
            fig, ax = visualizer.plot_current_curve(
                maturities, zero_rates,
                title=f"Yield Curve - Step {st.session_state.current_step}/{steps}"
            )
            
            # Display the plot
            plot_placeholder.pyplot(fig)
            
            # Display step info
            step_info.write(f"### Step {st.session_state.current_step}/{steps}")
            
            # Check for arbitrage opportunities
            curve_arbitrage = arbitrage_detector.detect_curve_arbitrage(market.yield_curve)
            cross_arbitrage = arbitrage_detector.check_cross_instrument_arbitrage(market)
            
            # Display arbitrage info
            with arbitrage_placeholder.container():
                if curve_arbitrage:
                    st.warning(f"Found {len(curve_arbitrage)} curve arbitrage opportunities!")
                    for opp in curve_arbitrage:
                        st.write(f"""
                        **Curve Arbitrage**: Between {opp['maturity_short']}y and {opp['maturity_long']}y maturities  
                        Potential profit: {opp['potential']:.2%}
                        """)
                
                if cross_arbitrage:
                    st.warning(f"Found {len(cross_arbitrage)} cross-instrument arbitrage opportunities!")
                    for opp in cross_arbitrage:
                        st.write(f"""
                        **Instrument Arbitrage**: Between {opp['instrument1']} and {opp['instrument2']}  
                        YTMs: {opp['ytm1']:.2%} vs {opp['ytm2']:.2%}  
                        Potential profit: {opp['potential']:.2%}
                        """)
                
                if not curve_arbitrage and not cross_arbitrage:
                    st.success("No arbitrage opportunities detected in this step.")
            
            # Display instrument info
            with instrument_placeholder.container():
                # Bank bills info
                if market.bank_bills:
                    st.write("**Bank Bills:**")
                    bill_data = {
                        "Maturity": [bill.get_maturity() for bill in market.bank_bills],
                        "YTM": [f"{bill.get_ytm():.2%}" for bill in market.bank_bills],
                        "Price": [f"${bill.get_price():.2f}" for bill in market.bank_bills]
                    }
                    st.dataframe(pd.DataFrame(bill_data))
                
                # Bonds info
                if market.bonds:
                    st.write("**Bonds:**")
                    bond_data = {
                        "Maturity": [bond.get_maturity() for bond in market.bonds],
                        "Coupon": [f"{bond.get_coupon_rate():.2%}" for bond in market.bonds],
                        "YTM": [f"{bond.get_ytm():.2%}" for bond in market.bonds],
                        "Price": [f"${bond.get_price():.2f}" for bond in market.bonds]
                    }
                    st.dataframe(pd.DataFrame(bond_data))
        else:
            st.warning("No valid curve data available for this step.")
    else:
        st.info("Click 'Run Next Step' to start the simulation.")

elif simulation_mode == "Auto-Run":
    # Check if we need to pause
    is_paused = st.sidebar.button("Pause Simulation")
    
    if is_paused:
        st.session_state.simulation_state = 'paused'
        st.rerun()
    
    # Check if we need to start/resume
    if st.session_state.simulation_state == 'paused' or st.session_state.simulation_state == 'initial':
        is_resumed = st.sidebar.button("Start/Resume Simulation")
        if is_resumed:
            st.session_state.simulation_state = 'running'
            st.rerun()
    
    # Auto-run simulation
    if st.session_state.simulation_state == 'running' and st.session_state.current_step < steps:
        # Run one step
        run_simulation_step()
        
        # Get the current yield curve
        maturities, discount_factors = market.yield_curve.get_zero_curve()
        
        if len(maturities) > 0 and len(discount_factors) > 0:
            # Convert discount factors to zero rates for display
            zero_rates = []
            for i, mat in enumerate(maturities):
                if mat > 0:
                    zero_rates.append(-np.log(discount_factors[i])/mat)
                else:
                    zero_rates.append(0)
            
            # Add the curve to the visualizer history
            visualizer.add_curve(maturities, zero_rates)
            
            # Create the current curve plot
            fig, ax = visualizer.plot_current_curve(
                maturities, zero_rates,
                title=f"Yield Curve - Step {st.session_state.current_step}/{steps}"
            )
            
            # Display the plot
            plot_placeholder.pyplot(fig)
            
            # Display step info
            step_info.write(f"### Step {st.session_state.current_step}/{steps}")
            
            # Check for arbitrage opportunities
            curve_arbitrage = arbitrage_detector.detect_curve_arbitrage(market.yield_curve)
            cross_arbitrage = arbitrage_detector.check_cross_instrument_arbitrage(market)
            
            # Display arbitrage info
            with arbitrage_placeholder.container():
                if curve_arbitrage:
                    st.warning(f"Found {len(curve_arbitrage)} curve arbitrage opportunities!")
                    for opp in curve_arbitrage:
                        st.write(f"""
                        **Curve Arbitrage**: Between {opp['maturity_short']}y and {opp['maturity_long']}y maturities  
                        Potential profit: {opp['potential']:.2%}
                        """)
                
                if cross_arbitrage:
                    st.warning(f"Found {len(cross_arbitrage)} cross-instrument arbitrage opportunities!")
                    for opp in cross_arbitrage:
                        st.write(f"""
                        **Instrument Arbitrage**: Between {opp['instrument1']} and {opp['instrument2']}  
                        YTMs: {opp['ytm1']:.2%} vs {opp['ytm2']:.2%}  
                        Potential profit: {opp['potential']:.2%}
                        """)
                
                if not curve_arbitrage and not cross_arbitrage:
                    st.success("No arbitrage opportunities detected in this step.")
            
            # Display instrument info
            with instrument_placeholder.container():
                # Bank bills info
                if market.bank_bills:
                    st.write("**Bank Bills:**")
                    bill_data = {
                        "Maturity": [bill.get_maturity() for bill in market.bank_bills],
                        "YTM": [f"{bill.get_ytm():.2%}" for bill in market.bank_bills],
                        "Price": [f"${bill.get_price():.2f}" for bill in market.bank_bills]
                    }
                    st.dataframe(pd.DataFrame(bill_data))
                
                # Bonds info
                if market.bonds:
                    st.write("**Bonds:**")
                    bond_data = {
                        "Maturity": [bond.get_maturity() for bond in market.bonds],
                        "Coupon": [f"{bond.get_coupon_rate():.2%}" for bond in market.bonds],
                        "YTM": [f"{bond.get_ytm():.2%}" for bond in market.bonds],
                        "Price": [f"${bond.get_price():.2f}" for bond in market.bonds]
                    }
                    st.dataframe(pd.DataFrame(bond_data))
        else:
            st.warning("No valid curve data available for this step.")
        
        # Add auto-rerun with delay
        if st.session_state.simulation_state == 'running' and st.session_state.current_step < steps:
            time.sleep(update_speed / 1000)
            st.rerun()
    
    if st.session_state.current_step >= steps:
        st.success("Simulation completed!")

# Show final animation if simulation is finished
if st.session_state.simulation_state == 'finished':
    st.success("Simulation completed!")
    
    # Check if we have enough curve history to create an animation
    if len(visualizer.curve_history) > 1:
        st.subheader("Animation of Yield Curve Evolution")
        
        # Create grid visualization as a fallback option
        grid_fig = visualizer.create_static_animation_grid(max_curves=9)
        st.pyplot(grid_fig)
        
        try:
            # Create animation
            ani = visualizer.create_animation(interval=update_speed)
            
            if ani:
                # Save animation to a temporary file
                animation_file = "yield_curve_animation.gif"
                
                try:
                    # Use a more direct approach to save frames
                    plt.rcParams['animation.writer'] = 'pillow'
                    ani.save(animation_file, writer='pillow', fps=2)
                    
                    # Display the animation
                    try:
                        with open(animation_file, "rb") as file:
                            btn = st.download_button(
                                label="Download Animation",
                                data=file,
                                file_name="yield_curve_animation.gif",
                                mime="image/gif"
                            )
                        
                        # Display image
                        st.image(animation_file)
                    except Exception as e:
                        st.error(f"Error displaying animation: {e}")
                        
                except Exception as e:
                    st.error(f"Error saving animation: {e}")
        except Exception as e:
            st.error(f"Error creating animation: {e}")
    else:
        st.warning("Not enough data to create an animation. Run more simulation steps.")
        
        # Show the last frame if available
        if visualizer.curve_history:
            # Plot the first and last curves on the same graph
            first_maturities, first_rates = visualizer.curve_history[0]
            print(visualizer.curve_history)
            last_maturities, last_rates = visualizer.curve_history[-1]
            fig, ax = plt.subplots(figsize=VISUALIZATION['figsize'])
            ax.plot(first_maturities, first_rates, 'o-', label="First Yield Curve")
            ax.plot(last_maturities, last_rates, 's--', label="Final Yield Curve")
            ax.set_title("First vs Final Yield Curve")
            ax.set_xlabel("Maturity (Years)")
            ax.set_ylabel("Zero Rate")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            

            #             fig, ax = visualizer.plot_current_curve(
            #     maturities, zero_rates,
            # ) first and last
            
            # # Display the plot
            # plot_placeholder.pyplot(fig)