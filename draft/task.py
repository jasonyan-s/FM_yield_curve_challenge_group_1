# task.py
import streamlit as st
import datetime as dt
import time
import matplotlib.pyplot as plt

from market_simulation import MarketSimulation, create_custom_market_simulation
from instruments import BankBill, Bond
from derivatives import ForwardRateAgreement, BondForward

# ---------------------- Streamlit App ------------s----------

def main():
    st.set_page_config(page_title="Financial Market Simulator", layout="wide")
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .price-up {color: green; font-weight: bold;}
        .price-down {color: red; font-weight: bold;}
        .big-number {font-size: 24px; font-weight: bold;}
        .card {
            padding: 20px;
            border-radius: 5px;
            background-color: #f8f9fa;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
        }
        .instrument-card {
            border-left: 4px solid #4c78a8;
            padding-left: 10px;
        }
        .arbitrage-opportunity {
            background-color: #fffacd;
            border-left: 4px solid #ffd700;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Dynamic Financial Market Simulator")

    # Create sidebar for simulation parameters
    st.sidebar.header("Simulation Parameters")
    
    # Yield curve parameters
    st.sidebar.subheader("Yield Curve Parameters")
    
    # Define default values
    default_values = {
        'rate_30d': 0.045, 
        'rate_60d': 0.047, 
        'rate_90d': 0.05, 
        'rate_180d': 0.053,
        'rate_1y': 0.056, 
        'rate_2y': 0.058, 
        'rate_5y': 0.062, 
        'rate_10y': 0.067,
        'bill_volatility': 0.5,
        'bond_volatility': 0.5,
        'fra_volatility': 0.7,
        'bond_forward_volatility': 0.8,
        'short_medium_correlation': 0.7,
        'medium_long_correlation': 0.6,
        'market_drift': 0.03
    }
    
    # Track parameter changes to detect when to reload yield curve
    previous_params = st.session_state.get('yield_curve_params', {})
    current_params = {}
    
    # Add Reset to Default Values button
    if st.sidebar.button("Reset to Default Values"):
        # Reset all parameters to default values
        for key in default_values:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state['reset_triggered'] = True
        st.rerun()
    
    # Check if reset was triggered and clear the flag
    if 'reset_triggered' in st.session_state:
        del st.session_state['reset_triggered']
        
    # Use session state to maintain values between reruns
    if 'rate_30d' not in st.session_state:
        st.session_state['rate_30d'] = default_values['rate_30d'] * 100
    if 'rate_60d' not in st.session_state:
        st.session_state['rate_60d'] = default_values['rate_60d'] * 100
    if 'rate_90d' not in st.session_state:
        st.session_state['rate_90d'] = default_values['rate_90d'] * 100
    if 'rate_180d' not in st.session_state:
        st.session_state['rate_180d'] = default_values['rate_180d'] * 100
    if 'rate_1y' not in st.session_state:
        st.session_state['rate_1y'] = default_values['rate_1y'] * 100
    if 'rate_2y' not in st.session_state:
        st.session_state['rate_2y'] = default_values['rate_2y'] * 100
    if 'rate_5y' not in st.session_state:
        st.session_state['rate_5y'] = default_values['rate_5y'] * 100
    if 'rate_10y' not in st.session_state:
        st.session_state['rate_10y'] = default_values['rate_10y'] * 100
    
    # Using key parameter to trigger callback on change
    current_params['rate_30d'] = st.sidebar.slider("30-day Rate (%)", 1.0, 10.0, st.session_state['rate_30d'], 0.1, key='rate_30d_slider') / 100
    current_params['rate_60d'] = st.sidebar.slider("60-day Rate (%)", 1.0, 10.0, st.session_state['rate_60d'], 0.1, key='rate_60d_slider') / 100
    current_params['rate_90d'] = st.sidebar.slider("90-day Rate (%)", 1.0, 10.0, st.session_state['rate_90d'], 0.1, key='rate_90d_slider') / 100
    current_params['rate_180d'] = st.sidebar.slider("180-day Rate (%)", 1.0, 10.0, st.session_state['rate_180d'], 0.1, key='rate_180d_slider') / 100
    current_params['rate_1y'] = st.sidebar.slider("1-year Rate (%)", 1.0, 10.0, st.session_state['rate_1y'], 0.1, key='rate_1y_slider') / 100
    current_params['rate_2y'] = st.sidebar.slider("2-year Rate (%)", 1.0, 10.0, st.session_state['rate_2y'], 0.1, key='rate_2y_slider') / 100
    current_params['rate_5y'] = st.sidebar.slider("5-year Rate (%)", 1.0, 10.0, st.session_state['rate_5y'], 0.1, key='rate_5y_slider') / 100
    current_params['rate_10y'] = st.sidebar.slider("10-year Rate (%)", 1.0, 10.0, st.session_state['rate_10y'], 0.1, key='rate_10y_slider') / 100
    
    # Update session state for next time
    st.session_state['rate_30d'] = current_params['rate_30d'] * 100
    st.session_state['rate_60d'] = current_params['rate_60d'] * 100
    st.session_state['rate_90d'] = current_params['rate_90d'] * 100
    st.session_state['rate_180d'] = current_params['rate_180d'] * 100
    st.session_state['rate_1y'] = current_params['rate_1y'] * 100
    st.session_state['rate_2y'] = current_params['rate_2y'] * 100
    st.session_state['rate_5y'] = current_params['rate_5y'] * 100
    st.session_state['rate_10y'] = current_params['rate_10y'] * 100
    
    # Check if any parameter has changed
    if previous_params != current_params and previous_params:
        # Show alert about parameter change
        st.sidebar.warning("⚠️ Yield curve parameters changed! Market will update automatically.")
        
        # Save current parameters for next comparison
        st.session_state.yield_curve_params = current_params
        
        # Reset the market with new yield curve parameters
        if 'market_sim' in st.session_state:
            # Reset rates to new values
            for i, bill in enumerate(st.session_state.market_sim.bank_bills):
                maturity_days = bill.maturity_days
                if maturity_days == 30:
                    bill.update_yield(current_params['rate_30d'])
                elif maturity_days == 60:
                    bill.update_yield(current_params['rate_60d'])
                elif maturity_days == 90:
                    bill.update_yield(current_params['rate_90d'])
                elif maturity_days == 180:
                    bill.update_yield(current_params['rate_180d'])
            
            for i, bond in enumerate(st.session_state.market_sim.bonds):
                if bond.maturity_years == 1:
                    bond.update_ytm(current_params['rate_1y'])
                elif bond.maturity_years == 2:
                    bond.update_ytm(current_params['rate_2y'])
                elif bond.maturity_years == 5:
                    bond.update_ytm(current_params['rate_5y'])
                elif bond.maturity_years == 10:
                    bond.update_ytm(current_params['rate_10y'])
                    
            # Update yield curve
            st.session_state.market_sim.yield_curve.update_curve()
            
            # Update derivatives based on new underlying prices
            for fra in st.session_state.market_sim.fras:
                fra.update_forward_rate(fra.calculate_theoretical_forward_rate())
            
            for bf in st.session_state.market_sim.bond_forwards:
                bf.update_forward_yield(bf.calculate_theoretical_forward_yield())
    else:
        # Just initialize the yield_curve_params with current parameters if this is the first run
        st.session_state.yield_curve_params = current_params
    
    # Volatility parameters
    st.sidebar.subheader("Volatility Parameters")
    
    # Track previous volatility parameters for comparison
    previous_vol_params = st.session_state.get('volatility_params', {})
    current_vol_params = {}
    
    if 'bill_volatility' not in st.session_state:
        st.session_state['bill_volatility'] = default_values['bill_volatility']
    if 'bond_volatility' not in st.session_state:
        st.session_state['bond_volatility'] = default_values['bond_volatility']
    if 'fra_volatility' not in st.session_state:
        st.session_state['fra_volatility'] = default_values['fra_volatility']
    if 'bond_forward_volatility' not in st.session_state:
        st.session_state['bond_forward_volatility'] = default_values['bond_forward_volatility']
    
    current_vol_params['bill_volatility'] = st.sidebar.slider("Bank Bill Volatility", 0.1, 1.0, st.session_state['bill_volatility'], 0.1, key='bill_volatility_slider')
    current_vol_params['bond_volatility'] = st.sidebar.slider("Bond Volatility", 0.1, 1.0, st.session_state['bond_volatility'], 0.1, key='bond_volatility_slider')
    current_vol_params['fra_volatility'] = st.sidebar.slider("FRA Volatility", 0.1, 1.5, st.session_state['fra_volatility'], 0.1, key='fra_volatility_slider')
    current_vol_params['bond_forward_volatility'] = st.sidebar.slider("Bond Forward Volatility", 0.1, 1.5, st.session_state['bond_forward_volatility'], 0.1, key='bond_forward_volatility_slider')
    
    st.session_state['bill_volatility'] = current_vol_params['bill_volatility']
    st.session_state['bond_volatility'] = current_vol_params['bond_volatility']
    st.session_state['fra_volatility'] = current_vol_params['fra_volatility']
    st.session_state['bond_forward_volatility'] = current_vol_params['bond_forward_volatility']
    
    # Check if volatility parameters have changed
    if previous_vol_params != current_vol_params and previous_vol_params:
        st.sidebar.warning("⚠️ Volatility parameters changed! Market behavior will be affected.")
        
    # Store current parameters for next comparison
    st.session_state.volatility_params = current_vol_params
    
    # Correlation parameters
    st.sidebar.subheader("Correlation Parameters")
    
    # Track previous correlation parameters for comparison
    previous_corr_params = st.session_state.get('correlation_params', {})
    current_corr_params = {}
    
    if 'short_medium_correlation' not in st.session_state:
        st.session_state['short_medium_correlation'] = default_values['short_medium_correlation']
    if 'medium_long_correlation' not in st.session_state:
        st.session_state['medium_long_correlation'] = default_values['medium_long_correlation']
    
    current_corr_params['short_medium_correlation'] = st.sidebar.slider("Short-Medium Correlation", -1.0, 1.0, st.session_state['short_medium_correlation'], 0.1, key='short_medium_correlation_slider')
    current_corr_params['medium_long_correlation'] = st.sidebar.slider("Medium-Long Correlation", -1.0, 1.0, st.session_state['medium_long_correlation'], 0.1, key='medium_long_correlation_slider')
    
    st.session_state['short_medium_correlation'] = current_corr_params['short_medium_correlation']
    st.session_state['medium_long_correlation'] = current_corr_params['medium_long_correlation']
    
    # Check if correlation parameters have changed
    if previous_corr_params != current_corr_params and previous_corr_params:
        st.sidebar.warning("⚠️ Correlation parameters changed! This will affect how instruments move together.")
    
    # Store current parameters for next comparison
    st.session_state.correlation_params = current_corr_params
    
    # Market update behavior
    st.sidebar.subheader("Market Update Behavior")
    
    # Track previous market drift parameter for comparison
    previous_drift = st.session_state.get('market_drift_param', None)
    
    if 'market_drift' not in st.session_state:
        st.session_state['market_drift'] = default_values['market_drift'] * 100
    
    market_drift = st.sidebar.slider("Market Drift (%/year)", -5.0, 5.0, st.session_state['market_drift'], 0.1, key='market_drift_slider') / 100
    
    st.session_state['market_drift'] = market_drift * 100
    
    # Check if market drift parameter has changed
    if previous_drift is not None and previous_drift != market_drift:
        st.sidebar.warning("⚠️ Market drift parameter changed! This will affect the long-term trend of rates.")
    
    # Store current parameter for next comparison
    st.session_state.market_drift_param = market_drift
    
    # Initialize or update simulation
    if 'market_sim' not in st.session_state or st.sidebar.button("Reset Simulation"):
        with st.spinner("Initializing market simulation..."):
            # Create custom market simulation with user parameters
            # Assign rates from current_params before calling create_custom_market_simulation
            rate_30d = current_params['rate_30d']
            rate_60d = current_params['rate_60d']
            rate_90d = current_params['rate_90d']
            rate_180d = current_params['rate_180d']
            rate_1y = current_params['rate_1y']
            rate_2y = current_params['rate_2y']
            rate_5y = current_params['rate_5y']
            rate_10y = current_params['rate_10y']
            st.session_state.market_sim = create_custom_market_simulation(
                rate_30d=rate_30d,
                rate_60d=rate_60d,
                rate_90d=rate_90d,
                rate_180d=rate_180d,
                rate_1y=rate_1y,
                rate_2y=rate_2y,
                rate_5y=rate_5y,
                rate_10y=rate_10y
            )
            st.session_state.volatility = st.session_state['bill_volatility']  # Default volatility
            st.session_state.update_count = 0
            st.session_state.price_history = {
                'bank_bills': {i: [] for i in range(len(st.session_state.market_sim.bank_bills))},
                'bonds': {i: [] for i in range(len(st.session_state.market_sim.bonds))},
                'fras': {i: [] for i in range(len(st.session_state.market_sim.fras))},
                'bond_forwards': {i: [] for i in range(len(st.session_state.market_sim.bond_forwards))},
            }
            st.session_state.yield_history = []
            maturities = st.session_state.market_sim.yield_curve.maturities
            yields = st.session_state.market_sim.yield_curve.yields
            st.session_state.yield_history.append((maturities, yields))
            st.session_state.timestamps = []
            st.session_state.start_time = dt.datetime.now()
            # Initialize price change tracking
            st.session_state.previous_prices = {
                'bank_bills': [bill.price for bill in st.session_state.market_sim.bank_bills],
                'bonds': [bond.price for bond in st.session_state.market_sim.bonds],
                'fras': [fra.price for fra in st.session_state.market_sim.fras],
                'bond_forwards': [bf.price for bf in st.session_state.market_sim.bond_forwards],
            }
            # Initialize cumulative arbitrage tracking with all instrument types
            st.session_state.arbitrage_history = {
                "bank_bill": [],
                "bond": [],
                "fra": [],
                "bond_forward": []
            }
    
    # Main content
    st.markdown("""
    FINMA Module 3 Financial Modelling Challenge Task - Jason Yan, Nathaniel Van Beelen, Serena Chui, Aaryan Gandhi, Molly Henry, Daniel Nemani, with the assistance of Claude 3.7 Sonnet.
    """)
    
    # Create the layout
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.subheader("Market Controls")
        
        with st.container():
            volatility = st.slider("Market Volatility", 
                                  min_value=0.1, 
                                  max_value=1.0, 
                                  value=st.session_state.volatility,
                                  step=0.1,
                                  help="Higher volatility = larger price movements")
            st.session_state.volatility = volatility
            
            # Add a scale input for number of time steps
            num_time_steps = st.slider("Number of Time Steps", 
                                min_value=1, 
                                max_value=1000, 
                                value=1, 
                                step=1,
                                help="Number of market updates to perform at once")
        
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Update Market", use_container_width=True):
                    # Save previous prices before update
                    st.session_state.previous_prices = {
                        'bank_bills': [bill.price for bill in st.session_state.market_sim.bank_bills],
                        'bonds': [bond.price for bond in st.session_state.market_sim.bonds],
                        'fras': [fra.price for fra in st.session_state.market_sim.fras],
                        'bond_forwards': [bf.price for bf in st.session_state.market_sim.bond_forwards],
                    }
                    
                    # Perform multiple updates based on the num_time_steps slider
                    with st.spinner(f"Performing {num_time_steps} market updates..."):
                        for _ in range(num_time_steps):
                            # Update the market with custom volatilities
                            st.session_state.market_sim.update_market(
                                base_volatility=volatility,
                                bill_vol_factor=st.session_state['bill_volatility'],
                                bond_vol_factor=st.session_state['bond_volatility'],
                                fra_vol_factor=st.session_state['fra_volatility'],
                                bf_vol_factor=st.session_state['bond_forward_volatility'],
                                drift=market_drift,
                                short_medium_corr=st.session_state['short_medium_correlation'],
                                medium_long_corr=st.session_state['medium_long_correlation']
                            )
                            st.session_state.update_count += 1
                            current_time = dt.datetime.now()
                            st.session_state.timestamps.append(current_time)
                            
                            # Update price history
                            for i, bill in enumerate(st.session_state.market_sim.bank_bills):
                                st.session_state.price_history['bank_bills'][i].append(bill.price)
                            for i, bond in enumerate(st.session_state.market_sim.bonds):
                                st.session_state.price_history['bonds'][i].append(bond.price)
                            for i, fra in enumerate(st.session_state.market_sim.fras):
                                st.session_state.price_history['fras'][i].append(fra.price)
                            for i, bf in enumerate(st.session_state.market_sim.bond_forwards):
                                st.session_state.price_history['bond_forwards'][i].append(bf.price)
                            
                            # Track arbitrage opportunities
                            opportunities = st.session_state.market_sim.get_arbitrage_opportunities()
                            
                            # Add update count to each opportunity for tracking when it occurred
                            for opp in opportunities["fra"]:
                                opp["update_count"] = st.session_state.update_count
                                opp["timestamp"] = current_time.strftime("%H:%M:%S")
                                st.session_state.arbitrage_history["fra"].append(opp)
                            
                            for opp in opportunities["bond_forward"]:
                                opp["update_count"] = st.session_state.update_count
                                opp["timestamp"] = current_time.strftime("%H:%M:%S")
                                st.session_state.arbitrage_history["bond_forward"].append(opp)
                            
                            # Add current yield curve snapshot if it's the last update
                            if _ == num_time_steps - 1:
                                maturities = st.session_state.market_sim.yield_curve.maturities
                                yields = st.session_state.market_sim.yield_curve.yields
                                st.session_state.yield_history.append((maturities, yields))
            
            with col2:
                if st.button("Reset Market", use_container_width=True):
                    # Reset just the market prices without changing structure
                    with st.spinner("Resetting market prices..."):
                        # Define initial rates based on sidebar parameters
                        # Using the specific tenor rates directly
                        rate_30d = st.session_state['rate_30d'] / 100
                        rate_60d = st.session_state['rate_60d'] / 100
                        rate_90d = st.session_state['rate_90d'] / 100
                        rate_180d = st.session_state['rate_180d'] / 100
                        rate_1y = st.session_state['rate_1y'] / 100
                        rate_2y = st.session_state['rate_2y'] / 100
                        rate_5y = st.session_state['rate_5y'] / 100
                        rate_10y = st.session_state['rate_10y'] / 100
                        
                        # Reset rates to initial values
                        for i, bill in enumerate(st.session_state.market_sim.bank_bills):
                            maturity_days = bill.maturity_days
                            if maturity_days == 30:
                                bill.update_yield(rate_30d)
                            elif maturity_days == 60:
                                bill.update_yield(rate_60d)
                            elif maturity_days == 90:
                                bill.update_yield(rate_90d)
                            elif maturity_days == 180:
                                bill.update_yield(rate_180d)
                        
                        for i, bond in enumerate(st.session_state.market_sim.bonds):
                            if bond.maturity_years == 1:
                                bond.update_ytm(rate_1y)
                            elif bond.maturity_years == 2:
                                bond.update_ytm(rate_2y)
                            elif bond.maturity_years == 5:
                                bond.update_ytm(rate_5y)
                            elif bond.maturity_years == 10:
                                bond.update_ytm(rate_10y)
                                
                        st.session_state.market_sim.yield_curve.update_curve()
                        
                        # Reset derivatives based on new underlying prices
                        for fra in st.session_state.market_sim.fras:
                            fra.update_forward_rate(fra.calculate_theoretical_forward_rate())
                        
                        for bf in st.session_state.market_sim.bond_forwards:
                            bf.update_forward_yield(bf.calculate_theoretical_forward_yield())
                            
                        # Update session state
                        st.session_state.previous_prices = {
                            'bank_bills': [bill.price for bill in st.session_state.market_sim.bank_bills],
                            'bonds': [bond.price for bond in st.session_state.market_sim.bonds],
                            'fras': [fra.price for fra in st.session_state.market_sim.fras],
                            'bond_forwards': [bf.price for bf in st.session_state.market_sim.bond_forwards],
                        }
            
            auto_update = st.checkbox("Auto-update Market")
            update_interval = st.slider("Update Interval (seconds)", 1, 10, 3, disabled=not auto_update)
            
            st.markdown(f"""
            <div style="text-align: center">
                <p>Market Updates: <span class="big-number">{st.session_state.update_count}</span></p>
                <p>Running for: <span>{(dt.datetime.now() - st.session_state.start_time).seconds} seconds</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Market Summary Section
            st.subheader("Market Summary")
            
            # Display arbitrage opportunities summary
            opportunities = st.session_state.market_sim.get_arbitrage_opportunities()
            total_opportunities = (len(opportunities["bank_bill"]) + len(opportunities["bond"]) + 
                                  len(opportunities["fra"]) + len(opportunities["bond_forward"]))
            
            if total_opportunities > 0:
                st.markdown(f"""
                <div style="text-align: center">
                    <p>Arbitrage Opportunities: <span class="big-number" style="color: gold;">{total_opportunities}</span></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align: center">
                    <p>Arbitrage Opportunities: <span class="big-number">0</span></p>
                </div>
                """, unsafe_allow_html=True)
    
    with right_col:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Yield Curve", "Price History", "Rate History"])
        
        with tab1:
            st.subheader("Dynamic Yield Curve")
            # Plot the current yield curve
            st.pyplot(st.session_state.market_sim.yield_curve.plot())
            
            # Add yield curve animation if we have history
            if len(st.session_state.yield_history) > 0:
                st.subheader("Yield Curve Evolution")
                # Create an animated plot of the yield curve over time
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot the first curve
                first_maturities, first_yields = st.session_state.yield_history[0]
                ax.plot(first_maturities, [y * 100 for y in first_yields], 'o-', alpha=0.5, color='#333333')
                
                # Plot the latest curve
                last_maturities, last_yields = st.session_state.yield_history[-1]
                ax.plot(last_maturities, [y * 100 for y in last_yields], 'o-', linewidth=2, color='blue')
                
                max_yield = max(max([y * 100 for y in first_yields] or [0]), max([y * 100 for y in last_yields] or [0]))
                min_yield = min(min([y * 100 for y in first_yields] or [0]), min([y * 100 for y in last_yields] or [0]))
                ax.set_ylim([min_yield - 0.5, max_yield + 0.5])
                # Annotate data points with yield and maturity
                for x, y in zip(last_maturities, [y * 100 for y in last_yields]):
                    ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color='blue')
                for x, y in zip(first_maturities, [y * 100 for y in first_yields]):
                    ax.annotate(f"{y:.2f}%", (x, y), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=8, color='#333333')

                ax.set_xlabel('Maturity (years)')
                ax.set_ylabel('Yield (%)')
                ax.set_title('Yield Curve Evolution')
                ax.grid(True)
                ax.legend(['Initial', 'Current'])
                
                st.pyplot(fig)
        
        with tab2:
            # Create price history charts for each instrument type
            if len(st.session_state.timestamps) > 1:
                instruments = st.radio(
                    "Select Instrument Type",
                    ["Bank Bills", "Bonds", "Forward Rate Agreements", "Bond Forwards"],
                    horizontal=True
                )
                
                if instruments == "Bank Bills":
                    # Plot bank bill price histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, history in st.session_state.price_history['bank_bills'].items():
                        if history:
                            bill = st.session_state.market_sim.bank_bills[i]
                            ax.plot(range(len(history)), history, '-', label=f"Bill {i+1} ({bill.maturity_days} days)")
                    
                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Price ($)')
                    ax.set_title('Bank Bill Price History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                
                elif instruments == "Bonds":
                    # Plot bond price histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, history in st.session_state.price_history['bonds'].items():
                        if history:
                            bond = st.session_state.market_sim.bonds[i]
                            ax.plot(range(len(history)), history, '-', label=f"Bond {i+1} ({bond.maturity_years} yrs)")
                    
                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Price ($)')
                    ax.set_title('Bond Price History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                
                elif instruments == "Forward Rate Agreements":
                    # Plot FRA price histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, history in st.session_state.price_history['fras'].items():
                        if history:
                            fra = st.session_state.market_sim.fras[i]
                            ax.plot(range(len(history)), history, '-', 
                                   label=f"FRA {i+1} (Bill: {fra.underlying_bill.maturity_days}d, Settle: {fra.settlement_days}d)")
                    
                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Price ($)')
                    ax.set_title('Forward Rate Agreement Price History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                
                elif instruments == "Bond Forwards":
                    # Plot bond forward price histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, history in st.session_state.price_history['bond_forwards'].items():
                        if history:
                            bf = st.session_state.market_sim.bond_forwards[i]
                            ax.plot(range(len(history)), history, '-', 
                                   label=f"BF {i+1} (Bond: {bf.underlying_bond.maturity_years}y, Settle: {bf.settlement_days}d)")
                    
                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Price ($)')
                    ax.set_title('Bond Forward Price History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.info("Run a few market updates to see price history charts")
        
        with tab3:
            # Create rate history charts for each instrument type
            if len(st.session_state.timestamps) > 1:
                instruments = st.radio(
                    "Select Instrument Type for Rate History",
                    ["Bank Bills", "Bonds", "Forward Rate Agreements", "Bond Forwards"],
                    horizontal=True,
                    key="rate_history_selector"
                )

                if instruments == "Bank Bills":
                    # Plot bank bill yield histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, bill in enumerate(st.session_state.market_sim.bank_bills):
                        # Extract yield rates from history
                        rates = []
                        for update_idx, price in enumerate(st.session_state.price_history['bank_bills'][i]):
                            # Temporarily create a bill with this price to get the yield
                            temp_bill = BankBill(
                                maturity_days=bill.maturity_days,
                                price=price
                            )
                            rates.append(temp_bill.yield_rate * 100)  # Convert to percentage

                        if rates:
                            ax.plot(range(len(rates)), rates, '-', label=f"Bill {i+1} ({bill.maturity_days} days)")

                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Yield Rate (%)')
                    ax.set_title('Bank Bill Yield History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)

                elif instruments == "Bonds":
                    # Plot bond YTM histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, history in st.session_state.price_history['bonds'].items():
                        if history:
                            bond = st.session_state.market_sim.bonds[i]
                            # Extract YTM rates from history
                            rates = []
                            for update_idx, price in enumerate(history):
                                # Temporarily create a bond with this price to get the YTM
                                temp_bond = Bond(
                                    face_value=bond.face_value,
                                    coupon_rate=bond.coupon_rate,
                                    maturity_years=bond.maturity_years,
                                    frequency=bond.frequency,
                                    price=price
                                )

                                rates.append(temp_bond.yield_to_maturity * 100)  # Convert to percentage

                            if rates:
                                ax.plot(range(len(rates)), rates, '-', label=f"Bond {i+1} ({bond.maturity_years} yrs)")
                    
                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Price ($)')
                    ax.set_title('Bond Price History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
                

                # fig, ax = plt.subplots(figsize=(10, 6))
        
        # data = self.history[instrument_type]
        # labels = []
        
        # for i, rates in data.items():
        #     if len(rates) > 1:  # Only plot if we have history
        #         if instrument_type == "bank_bills":
        #             label = f"Bill {i+1} ({self.bank_bills[i].maturity_days} days)"
        #         elif instrument_type == "bonds":
        #             label = f"Bond {i+1} ({self.bonds[i].maturity_years} years)"
        #         elif instrument_type == "fras":
        #             label = f"FRA {i+1} ({self.fras[i].settlement_days} days)"
        #         else:  # bond_forwards
        #             label = f"BF {i+1} ({self.bond_forwards[i].settlement_days} days)"
                
        #         ax.plot(rates, label=label)
        #         labels.append(label)
        
        # ax.set_xlabel('Market Updates')
        
        # if instrument_type in ["bank_bills", "bonds"]:
        #     ax.set_ylabel('Yield (%)')
        #     title = "Yield History"
        # elif instrument_type == "fras":
        #     ax.set_ylabel('Forward Rate (%)')
        #     title = "Forward Rate History"
        # else:
        #     ax.set_ylabel('Forward Yield (%)')
        #     title = "Forward Yield History"
            
        # # Convert to percentage for display
        # yticks = ax.get_yticks()
        # ax.set_yticks(yticks)
        # ax.set_yticklabels([f'{x*100:.2f}%' for x in yticks])
        
        # ax.set_title(title)
        # ax.grid(True)
        # ax.legend()
        
        # return fig

                    
                elif instruments == "Forward Rate Agreements":
                    # Plot FRA forward rate histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, fra in enumerate(st.session_state.market_sim.fras):
                        # Extract forward rates from history
                        rates = []
                        for update_idx, price in enumerate(st.session_state.price_history['fras'][i]):
                            # Temporarily create an FRA with this price to get the forward rate
                            temp_fra = ForwardRateAgreement(
                                underlying_bill=fra.underlying_bill,
                                settlement_days=fra.settlement_days,
                                price=price
                            )
                            rates.append(temp_fra.forward_rate * 100)  # Convert to percentage

                        if rates:
                            ax.plot(range(len(rates)), rates, '-',
                                    label=f"FRA {i+1} (Bill: {fra.underlying_bill.maturity_days}d, Settle: {fra.settlement_days}d)")

                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Forward Rate (%)')
                    ax.set_title('Forward Rate Agreement Rate History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)

                elif instruments == "Bond Forwards":
                    # Plot bond forward yield histories
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for i, bf in enumerate(st.session_state.market_sim.bond_forwards):
                        # Extract forward yields from history
                        rates = []
                        for update_idx, price in enumerate(st.session_state.price_history['bond_forwards'][i]):
                            # Temporarily create a bond forward with this price to get the forward yield
                            temp_bf = BondForward(
                                underlying_bond=bf.underlying_bond,
                                settlement_days=bf.settlement_days,
                                price=price
                            )
                            rates.append(temp_bf.forward_yield * 100)  # Convert to percentage

                        if rates:
                            ax.plot(range(len(rates)), rates, '-',
                                    label=f"BF {i+1} (Bond: {bf.underlying_bond.maturity_years}y, Settle: {bf.settlement_days}d)")

                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Forward Yield (%)')
                    ax.set_title('Bond Forward Yield History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.info("Run a few market updates to see rate history charts")



    # Market Data Section with enhanced dynamic display
    st.header("Live Market Data")
    
    # Create tabs for different instrument types
    tab1, tab2, tab3, tab4 = st.tabs(["Bank Bills", "Bonds", "Forward Rate Agreements", "Bond Forwards"])
    
    with tab1:
        st.subheader("Bank Bills")
        
        # Create columns for each bank bill for a card-like display
        cols = st.columns(len(st.session_state.market_sim.bank_bills))
        
        for i, bill in enumerate(st.session_state.market_sim.bank_bills):
            with cols[i]:
                # Determine price change direction
                prev_price = st.session_state.previous_prices['bank_bills'][i] if i < len(st.session_state.previous_prices['bank_bills']) else bill.price
                price_change = bill.price - prev_price
                price_class = "price-up" if price_change >= 0 else "price-down"
                price_arrow = "↑" if price_change > 0 else "↓" if price_change < 0 else ""
                
                # Format the price change
                price_change_formatted = f"{abs(price_change):.2f}" if price_change != 0 else "0.00"
                
                st.markdown(f"""
                <div class="card instrument-card">
                    <h4>Bank Bill {i+1}</h4>
                    <p>Maturity: <b>{bill.maturity_days} days</b></p>
                    <p>Price: <span class="{price_class}">${bill.price:.2f} {price_arrow}</span></p>
                    <p>Change: <span class="{price_class}">${price_change_formatted}</span></p>
                    <p>Yield: {bill.yield_rate*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("Bonds")
        
        # Create columns for each bond for a card-like display
        cols = st.columns(len(st.session_state.market_sim.bonds))
        
        for i, bond in enumerate(st.session_state.market_sim.bonds):
            with cols[i]:
                # Determine price change direction
                prev_price = st.session_state.previous_prices['bonds'][i] if i < len(st.session_state.previous_prices['bonds']) else bond.price
                price_change = bond.price - prev_price
                price_class = "price-up" if price_change >= 0 else "price-down"
                price_arrow = "↑" if price_change > 0 else "↓" if price_change < 0 else ""
                
                # Format the price change
                price_change_formatted = f"{abs(price_change):.2f}" if price_change != 0 else "0.00"
                
                st.markdown(f"""
                <div class="card instrument-card">
                    <h4>Bond {i+1}</h4>
                    <p>Maturity: <b>{bond.maturity_years} years</b></p>
                    <p>Coupon: {bond.coupon_rate*100:.2f}%</p>
                    <p>Price: <span class="{price_class}">${bond.price:.2f} {price_arrow}</span></p>
                    <p>Change: <span class="{price_class}">${price_change_formatted}</span></p>
                    <p>YTM: {bond.yield_to_maturity*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.subheader("Forward Rate Agreements (FRAs)")
        
        # Check if any FRAs have arbitrage opportunities
        opportunities = st.session_state.market_sim.get_arbitrage_opportunities()
        fra_opportunities = {opp["instrument"].split()[1]: opp for opp in opportunities["fra"]}
        
        # Create columns for each FRA for a card-like display
        cols = st.columns(len(st.session_state.market_sim.fras))
        
        for i, fra in enumerate(st.session_state.market_sim.fras):
            with cols[i]:
                               # Determine price change direction
                prev_price = st.session_state.previous_prices['fras'][i] if i < len(st.session_state.previous_prices['fras']) else fra.price
                price_change = fra.price - prev_price
                price_class = "price-up" if price_change >= 0 else "price-down"
                price_arrow = "↑" if price_change > 0 else "↓" if price_change < 0 else ""
                
                # Format the price change
                price_change_formatted = f"{abs(price_change):.2f}" if price_change != 0 else "0.00"
                
                # Check if this FRA has an arbitrage opportunity
                has_arbitrage = str(i+1) in fra_opportunities
                card_class = "card instrument-card arbitrage-opportunity" if has_arbitrage else "card instrument-card"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>FRA {i+1} {' 🔶 ARBITRAGE' if has_arbitrage else ''}</h4>
                    <p>Underlying Bill: <b>{fra.underlying_bill.maturity_days} days</b></p>
                    <p>Settlement: <b>{fra.settlement_days} days</b></p>
                    <p>Price: <span class="{price_class}">${fra.price:.2f} {price_arrow}</span></p>
                    <p>Change: <span class="{price_class}">${price_change_formatted}</span></p>
                    <p>Forward Rate: {fra.forward_rate*100:.2f}%</p>
                    <p>Theoretical Rate: {fra.calculate_theoretical_forward_rate()*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                if has_arbitrage:
                    opp = fra_opportunities[str(i+1)]
                    st.markdown(f"""
                    <div style="padding: 10px; background-color: #fff3cd; border-radius: 5px; margin-top: 5px;">
                        <p style="margin: 0; font-weight: bold;">
                            Action: {opp["action"]} 
                            (Profit: ${abs(opp["difference"]):.2f})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab4:
        st.subheader("Bond Forwards")
        
        # Check if any Bond Forwards have arbitrage opportunities
        opportunities = st.session_state.market_sim.get_arbitrage_opportunities()
        bf_opportunities = {opp["instrument"].split()[2]: opp for opp in opportunities["bond_forward"]}
        
        # Create columns for each Bond Forward for a card-like display
        cols = st.columns(len(st.session_state.market_sim.bond_forwards))
        
        for i, bf in enumerate(st.session_state.market_sim.bond_forwards):
            with cols[i]:
                # Determine price change direction
                prev_price = st.session_state.previous_prices['bond_forwards'][i] if i < len(st.session_state.previous_prices['bond_forwards']) else bf.price
                price_change = bf.price - prev_price
                price_class = "price-up" if price_change >= 0 else "price-down"
                price_arrow = "↑" if price_change > 0 else "↓" if price_change < 0 else ""
                
                # Format the price change
                price_change_formatted = f"{abs(price_change):.2f}" if price_change != 0 else "0.00"
                
                # Check if this Bond Forward has an arbitrage opportunity
                has_arbitrage = str(i+1) in bf_opportunities
                card_class = "card instrument-card arbitrage-opportunity" if has_arbitrage else "card instrument-card"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <h4>Bond Forward {i+1} {' 🔶 ARBITRAGE' if has_arbitrage else ''}</h4>
                    <p>Underlying Bond: <b>{bf.underlying_bond.maturity_years} years</b></p>
                    <p>Settlement: <b>{bf.settlement_days} days</b></p>
                    <p>Price: <span class="{price_class}">${bf.price:.2f} {price_arrow}</span></p>
                    <p>Change: <span class="{price_class}">${price_change_formatted}</span></p>
                    <p>Forward Yield: {bf.forward_yield*100:.2f}%</p>
                    <p>Theoretical Yield: {bf.calculate_theoretical_forward_yield()*100:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                if has_arbitrage:
                    opp = bf_opportunities[str(i+1)]
                    st.markdown(f"""
                    <div style="padding: 10px; background-color: #fff3cd; border-radius: 5px; margin-top: 5px;">
                        <p style="margin: 0; font-weight: bold;">
                            Action: {opp["action"]} 
                            (Profit: ${abs(opp["difference"]):.2f})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Arbitrage Opportunities Detailed Section
    st.header("Arbitrage Opportunities History Dashboard")

    # Check if we have any arbitrage history
    if not st.session_state.arbitrage_history["fra"] and not st.session_state.arbitrage_history["bond_forward"]:
        st.info("No arbitrage opportunities have been detected yet in the simulation.")
    else:
        # Create tabs for FRA and Bond Forward arbitrage histories
        arb_tab1, arb_tab2, arb_tab3, arb_tab4 = st.tabs(["All Opportunities", "FRA Opportunities", "Bond Forward Opportunities", "Multi-Instrument Opportunities"])
        
        with arb_tab1:
            st.subheader("All Arbitrage Opportunities")
            
            # Combine all arbitrage opportunities
            all_opps = []
            for opp in st.session_state.arbitrage_history["fra"]:
                all_opps.append({
                    "Update": opp["update_count"],
                    "Time": opp["timestamp"],
                    "Type": "FRA",
                    "Instrument": opp["instrument"],
                    "Description": opp["description"],
                    "Market Price": f"${opp['market_price']:.2f}",
                    "Theoretical Price": f"${opp['theoretical_price']:.2f}",
                    "Difference": f"${abs(opp['difference']):.2f}",
                    "Action": opp["action"],
                })
                
            for opp in st.session_state.arbitrage_history["bond_forward"]:
                all_opps.append({
                    "Update": opp["update_count"],
                    "Time": opp["timestamp"],
                    "Type": "Bond Forward",
                    "Instrument": opp["instrument"],
                    "Description": opp["description"],
                    "Market Price": f"${opp['market_price']:.2f}",
                    "Theoretical Price": f"${opp['theoretical_price']:.2f}",
                    "Difference": f"${abs(opp['difference']):.2f}",
                    "Action": opp["action"],
                })
            
            # Sort by update count (most recent first)
            all_opps = sorted(all_opps, key=lambda x: x["Update"], reverse=True)
            
            # Display as dataframe
            if all_opps:
                st.dataframe(
                    all_opps,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Action": st.column_config.TextColumn(
                            "Action",
                            help="Buy or Sell recommendation",
                            width="small",
                        ),
                        "Update": st.column_config.NumberColumn(
                            "Update",
                            help="Market update when opportunity was found",
                            format="%d",
                        ),
                        "Difference": st.column_config.TextColumn(
                            "Profit Potential",
                            help="Potential profit from arbitrage",
                        )
                    }
                )
            else:
                st.info("No arbitrage opportunities detected so far.")
        
        with arb_tab2:
            st.subheader("FRA Arbitrage Opportunities")
            
            # Prepare FRA opportunities for display
            fra_opps = []
            for opp in st.session_state.arbitrage_history["fra"]:
                fra_opps.append({
                    "Update": opp["update_count"],
                    "Time": opp["timestamp"],
                    "Instrument": opp["instrument"],
                    "Description": opp["description"],
                    "Market Price": f"${opp['market_price']:.2f}",
                    "Theoretical Price": f"${opp['theoretical_price']:.2f}",
                    "Difference": f"${abs(opp['difference']):.2f}",
                    "Action": opp["action"],
                })
            
            # Sort by update count (most recent first)
            fra_opps = sorted(fra_opps, key=lambda x: x["Update"], reverse=True)
            
            # Display as dataframe
            if fra_opps:
                st.dataframe(
                    fra_opps,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Action": st.column_config.TextColumn(
                            "Action",
                            help="Buy or Sell recommendation",
                            width="small",
                        ),
                    }
                )
            else:
                st.info("No FRA arbitrage opportunities detected so far.")
        
        with arb_tab3:
            st.subheader("Bond Forward Arbitrage Opportunities")
            
            # Prepare Bond Forward opportunities for display
            bf_opps = []
            for opp in st.session_state.arbitrage_history["bond_forward"]:
                bf_opps.append({
                    "Update": opp["update_count"],
                    "Time": opp["timestamp"],
                    "Instrument": opp["instrument"],
                    "Description": opp["description"],
                    "Market Price": f"${opp['market_price']:.2f}",
                    "Theoretical Price": f"${opp['theoretical_price']:.2f}",
                    "Difference": f"${abs(opp['difference']):.2f}",
                    "Action": opp["action"],
                })
            
            # Sort by update count (most recent first)
            bf_opps = sorted(bf_opps, key=lambda x: x["Update"], reverse=True)
            
            # Display as dataframe
            if bf_opps:
                st.dataframe(
                    bf_opps,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Action": st.column_config.TextColumn(
                            "Action",
                            help="Buy or Sell recommendation",
                            width="small",
                        ),
                    }
                )
            else:
                st.info("No Bond Forward arbitrage opportunities detected so far.")
        
        with arb_tab4:
            st.subheader("Multi-Instrument Arbitrage Opportunities")
            
            multi_opps = st.session_state.market_sim.get_multi_instrument_arbitrage()
            
            if (not multi_opps["butterfly"] and 
                not multi_opps["calendar_spread"] and 
                not multi_opps["triangulation"]):
                st.info("No multi-instrument arbitrage opportunities detected.")
            else:
                st.subheader("Butterfly Arbitrage")
                if multi_opps["butterfly"]:
                    st.dataframe(
                        multi_opps["butterfly"],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No butterfly arbitrage opportunities detected.")
                
                st.subheader("Calendar Spread Arbitrage")
                if multi_opps["calendar_spread"]:
                    st.dataframe(
                        multi_opps["calendar_spread"],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No calendar spread arbitrage opportunities detected.")
                
                st.subheader("Triangulation Arbitrage")
                if multi_opps["triangulation"]:
                    st.dataframe(
                        multi_opps["triangulation"],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No triangulation arbitrage opportunities detected.")

    # Display trading strategy explanation
    st.markdown("""
    <div style="text-align: center; padding: 15px; background-color: #f8f9fa; border-radius: 5px; margin-top: 20px;">
        <h4>Trading Strategy:</h4>
        <p><b>Buy</b> when market price is <b>below</b> theoretical price (undervalued)</p>
        <p><b>Sell</b> when market price is <b>above</b> theoretical price (overvalued)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-update functionality
    if 'auto_update' in locals() and auto_update:
        time.sleep(update_interval)
        # Save previous prices before update
        st.session_state.previous_prices = {
            'bank_bills': [bill.price for bill in st.session_state.market_sim.bank_bills],
            'bonds': [bond.price for bond in st.session_state.market_sim.bonds],
            'fras': [fra.price for fra in st.session_state.market_sim.fras],
            'bond_forwards': [bf.price for bf in st.session_state.market_sim.bond_forwards],
        }
        
        # Perform multiple updates based on the num_time_steps slider
        for _ in range(num_time_steps):
            # Update the market with custom volatilities
            st.session_state.market_sim.update_market(
                base_volatility=volatility,
                bill_vol_factor=st.session_state['bill_volatility'],
                bond_vol_factor=st.session_state['bond_volatility'],
                fra_vol_factor=st.session_state['fra_volatility'],
                bf_vol_factor=st.session_state['bond_forward_volatility'],
                drift=market_drift,
                short_medium_corr=st.session_state['short_medium_correlation'],
                medium_long_corr=st.session_state['medium_long_correlation']
            )
            st.session_state.update_count += 1
            current_time = dt.datetime.now()
            st.session_state.timestamps.append(current_time)
            
            # Update price history
            for i, bill in enumerate(st.session_state.market_sim.bank_bills):
                st.session_state.price_history['bank_bills'][i].append(bill.price)
            for i, bond in enumerate(st.session_state.market_sim.bonds):
                st.session_state.price_history['bonds'][i].append(bond.price)
            for i, fra in enumerate(st.session_state.market_sim.fras):
                st.session_state.price_history['fras'][i].append(fra.price)
            for i, bf in enumerate(st.session_state.market_sim.bond_forwards):
                st.session_state.price_history['bond_forwards'][i].append(bf.price)
        
        # Track arbitrage opportunities
        opportunities = st.session_state.market_sim.get_arbitrage_opportunities()
        
        # Add update count to each opportunity for tracking when it occurred
        for opp in opportunities["fra"]:
            opp["update_count"] = st.session_state.update_count
            opp["timestamp"] = current_time.strftime("%H:%M:%S")
            st.session_state.arbitrage_history["fra"].append(opp)
        

        
        for opp in opportunities["bond_forward"]:
            opp["update_count"] = st.session_state.update_count
            opp["timestamp"] = current_time.strftime("%H:%M:%S")
            st.session_state.arbitrage_history["bond_forward"].append(opp)
    
        # Add current yield curve snapshot
        maturities = st.session_state.market_sim.yield_curve.maturities
        yields = st.session_state.market_sim.yield_curve.yields
        st.session_state.yield_history.append((maturities, yields))
        
        st.rerun()



if __name__ == "__main__":
    main()