import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

# Set random seed for reproducibility
# np.random.seed(42)

# ---------------------- Financial Instruments ----------------------

@dataclass
class BankBill:
    """A short-term debt instrument"""
    face_value: float = 1000000  # $1M face value
    maturity_days: int = 90  # 90 day bank bill
    price: float = None
    yield_rate: float = None

    def __post_init__(self):
        if self.price is None and self.yield_rate is None:
            # Default to 5% yield
            self.yield_rate = 0.05
            self.price = self.calculate_price_from_yield(self.yield_rate)
        elif self.price is None:
            self.price = self.calculate_price_from_yield(self.yield_rate)
        elif self.yield_rate is None:
            self.yield_rate = self.calculate_yield_from_price(self.price)

    def calculate_price_from_yield(self, yield_rate: float) -> float:
        """Calculate price from yield"""
        return self.face_value / (1 + yield_rate * self.maturity_days / 365)

    def calculate_yield_from_price(self, price: float) -> float:
        """Calculate yield from price"""
        return (self.face_value / price - 1) * 365 / self.maturity_days

    def update_price(self, new_price: float):
        """Update price and recalculate yield"""
        self.price = new_price
        self.yield_rate = self.calculate_yield_from_price(new_price)

    def update_yield(self, new_yield: float):
        """Update yield and recalculate price"""
        self.yield_rate = new_yield
        self.price = self.calculate_price_from_yield(new_yield)

    def __str__(self) -> str:
        return f"BankBill(maturity={self.maturity_days} days, price=${self.price:.2f}, yield={self.yield_rate*100:.2f}%)"


@dataclass
class Bond:
    """A longer-term debt instrument"""
    face_value: float = 1000000  # $1M face value
    coupon_rate: float = 0.05  # 5% annual coupon rate
    maturity_years: float = 5  # 5-year bond
    frequency: int = 2  # Semi-annual coupon payments
    price: float = None
    yield_to_maturity: float = None

    def __post_init__(self):
        if self.price is None and self.yield_to_maturity is None:
            # Default to same YTM as coupon rate
            self.yield_to_maturity = self.coupon_rate
            self.price = self.calculate_price_from_ytm(self.yield_to_maturity)
        elif self.price is None:
            self.price = self.calculate_price_from_ytm(self.yield_to_maturity)
        elif self.yield_to_maturity is None:
            self.yield_to_maturity = self.calculate_ytm_from_price(self.price)

    def calculate_price_from_ytm(self, ytm: float) -> float:
        """Calculate bond price from yield to maturity using DCF"""
        periods = int(self.maturity_years * self.frequency)
        coupon_payment = self.face_value * self.coupon_rate / self.frequency
        r = ytm / self.frequency

        # Sum of discounted coupon payments
        pv_coupons = coupon_payment * (1 - (1 + r) ** -periods) / r

        # Present value of principal repayment
        pv_principal = self.face_value / (1 + r) ** periods

        return pv_coupons + pv_principal

    def calculate_ytm_from_price(self, price: float, tolerance: float = 1e-10) -> float:
        """Estimate YTM from price using numerical method"""
        # Initial guess - use coupon rate as starting point
        ytm_low, ytm_high = 0.0001, 1.0

        while ytm_high - ytm_low > tolerance:
            ytm_mid = (ytm_low + ytm_high) / 2
            price_mid = self.calculate_price_from_ytm(ytm_mid)

            if price_mid > price:
                ytm_low = ytm_mid
            else:
                ytm_high = ytm_mid

        return (ytm_low + ytm_high) / 2

    def update_price(self, new_price: float):
        """Update price and recalculate YTM"""
        self.price = new_price
        self.yield_to_maturity = self.calculate_ytm_from_price(new_price)

    def update_ytm(self, new_ytm: float):
        """Update YTM and recalculate price"""
        self.yield_to_maturity = new_ytm
        self.price = self.calculate_price_from_ytm(new_ytm)

    def __str__(self) -> str:
        return f"Bond(maturity={self.maturity_years} years, coupon={self.coupon_rate*100:.2f}%, " \
               f"price=${self.price:.2f}, YTM={self.yield_to_maturity*100:.2f}%)"


@dataclass
class ForwardRateAgreement:
    """A contract to buy or sell a bank bill on a future date"""
    underlying_bill: BankBill
    settlement_days: int = 180  # 180 days to settlement
    price: float = None
    forward_rate: float = None

    def __post_init__(self):
        if self.price is None and self.forward_rate is None:
            # Calculate theoretical forward rate
            self.forward_rate = self.calculate_theoretical_forward_rate()
            self.price = self.calculate_price_from_forward_rate(self.forward_rate)
        elif self.price is None:
            self.price = self.calculate_price_from_forward_rate(self.forward_rate)
        elif self.forward_rate is None:
            self.forward_rate = self.calculate_forward_rate_from_price(self.price)

    def calculate_theoretical_forward_rate(self) -> float:
        """Calculate the theoretical forward rate based on the yield curve"""
        # Simple implementation: add a small spread to current rate
        return self.underlying_bill.yield_rate + 0.005

    def calculate_price_from_forward_rate(self, forward_rate: float) -> float:
        """Calculate FRA price from forward rate"""
        future_bill_price = self.underlying_bill.face_value / (
            1 + forward_rate * self.underlying_bill.maturity_days / 365
        )
        # Discount back to present value
        discount_factor = 1 / (1 + self.underlying_bill.yield_rate * self.settlement_days / 365)
        return future_bill_price * discount_factor

    def calculate_forward_rate_from_price(self, price: float) -> float:
        """Calculate forward rate from FRA price"""
        # First, calculate future bill price by un-discounting the FRA price
        discount_factor = 1 / (1 + self.underlying_bill.yield_rate * self.settlement_days / 365)
        future_bill_price = price / discount_factor

        # Then calculate forward rate from future bill price
        return (self.underlying_bill.face_value / future_bill_price - 1) * 365 / self.underlying_bill.maturity_days

    def calculate_arbitrage_opportunity(self) -> Tuple[bool, float]:
        """Check if there's an arbitrage opportunity and return the potential profit"""
        theoretical_rate = self.calculate_theoretical_forward_rate()
        theoretical_price = self.calculate_price_from_forward_rate(theoretical_rate)

        diff = self.price - theoretical_price
        has_opportunity = abs(diff) > 1000  # Threshold for meaningful arbitrage

        return has_opportunity, diff

    def update_price(self, new_price: float):
        """Update price and recalculate forward rate"""
        self.price = new_price
        self.forward_rate = self.calculate_forward_rate_from_price(new_price)

    def update_forward_rate(self, new_rate: float):
        """Update forward rate and recalculate price"""
        self.forward_rate = new_rate
        self.price = self.calculate_price_from_forward_rate(new_rate)

    def __str__(self) -> str:
        return f"FRA(settlement={self.settlement_days} days, " \
               f"price=${self.price:.2f}, forward_rate={self.forward_rate*100:.2f}%)"


@dataclass
class BondForward:
    """A contract to buy or sell a bond on a future date"""
    underlying_bond: Bond
    settlement_days: int = 180  # 180 days to settlement
    price: float = None
    forward_yield: float = None

    def __post_init__(self):
        if self.price is None and self.forward_yield is None:
            # Calculate theoretical forward yield
            self.forward_yield = self.calculate_theoretical_forward_yield()
            self.price = self.calculate_price_from_forward_yield(self.forward_yield)
        elif self.price is None:
            self.price = self.calculate_price_from_forward_yield(self.forward_yield)
        elif self.forward_yield is None:
            self.forward_yield = self.calculate_forward_yield_from_price(self.price)

    def calculate_theoretical_forward_yield(self) -> float:
        """Calculate the theoretical forward yield based on the yield curve"""
        # Simple implementation: add a small spread to current YTM
        return self.underlying_bond.yield_to_maturity + 0.002

    def calculate_price_from_forward_yield(self, forward_yield: float) -> float:
        """Calculate forward price from forward yield"""
        # Calculate future bond price
        # Adjust maturity by settlement time
        adjusted_maturity = self.underlying_bond.maturity_years - (self.settlement_days / 365)

        # Create a temporary bond with adjusted maturity
        temp_bond = Bond(
            face_value=self.underlying_bond.face_value,
            coupon_rate=self.underlying_bond.coupon_rate,
            maturity_years=adjusted_maturity,
            frequency=self.underlying_bond.frequency,
            yield_to_maturity=forward_yield
        )

        future_bond_price = temp_bond.price

        # Discount back to present value
        discount_factor = 1 / (1 + self.underlying_bond.yield_to_maturity * self.settlement_days / 365)
        return future_bond_price * discount_factor

    def calculate_forward_yield_from_price(self, price: float) -> float:
        """Estimate forward yield from forward price"""
        # First, calculate future bond price by un-discounting the forward price
        discount_factor = 1 / (1 + self.underlying_bond.yield_to_maturity * self.settlement_days / 365)
        future_bond_price = price / discount_factor

        # Adjust maturity by settlement time
        adjusted_maturity = self.underlying_bond.maturity_years - (self.settlement_days / 365)

        # Create a temporary bond with adjusted maturity
        temp_bond = Bond(
            face_value=self.underlying_bond.face_value,
            coupon_rate=self.underlying_bond.coupon_rate,
            maturity_years=adjusted_maturity,
            frequency=self.underlying_bond.frequency,
            price=future_bond_price
        )

        return temp_bond.yield_to_maturity

    def calculate_arbitrage_opportunity(self) -> Tuple[bool, float]:
        """Check if there's an arbitrage opportunity and return the potential profit"""
        theoretical_yield = self.calculate_theoretical_forward_yield()
        theoretical_price = self.calculate_price_from_forward_yield(theoretical_yield)

        diff = self.price - theoretical_price
        has_opportunity = abs(diff) > 2000  # Threshold for meaningful arbitrage

        return has_opportunity, diff

    def update_price(self, new_price: float):
        """Update price and recalculate forward yield"""
        self.price = new_price
        self.forward_yield = self.calculate_forward_yield_from_price(new_price)

    def update_forward_yield(self, new_yield: float):
        """Update forward yield and recalculate price"""
        self.forward_yield = new_yield
        self.price = self.calculate_price_from_forward_yield(new_yield)

    def __str__(self) -> str:
        return f"BondForward(settlement={self.settlement_days} days, " \
               f"price=${self.price:.2f}, forward_yield={self.forward_yield*100:.2f}%)"


class YieldCurve:
    """A yield curve constructed from market instruments"""
    def __init__(self, bank_bills: List[BankBill], bonds: List[Bond]):
        self.bank_bills = sorted(bank_bills, key=lambda x: x.maturity_days)
        self.bonds = sorted(bonds, key=lambda x: x.maturity_years)
        self.maturities = []
        self.yields = []
        self.update_curve()

    def update_curve(self):
        """Update the yield curve points from current market instruments"""
        self.maturities = []
        self.yields = []

        # Add bank bill points
        for bill in self.bank_bills:
            self.maturities.append(bill.maturity_days / 365)
            self.yields.append(bill.yield_rate)

        # Add bond points
        for bond in self.bonds:
            self.maturities.append(bond.maturity_years)
            self.yields.append(bond.yield_to_maturity)

    def get_interpolated_rate(self, maturity_years: float) -> float:
        """Get interpolated yield rate for a specific maturity"""
        if maturity_years <= 0:
            return self.yields[0]
        if maturity_years >= self.maturities[-1]:
            return self.yields[-1]

        # Find the surrounding points
        for i in range(len(self.maturities) - 1):
            if self.maturities[i] <= maturity_years <= self.maturities[i + 1]:
                # Linear interpolation
                weight = (maturity_years - self.maturities[i]) / (self.maturities[i + 1] - self.maturities[i])
                return self.yields[i] + weight * (self.yields[i + 1] - self.yields[i])

    def plot(self):
        """Plot the yield curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.maturities, [y * 100 for y in self.yields], 'o-', linewidth=2)
        # Set y-axis to start at 0
        # Set y-axis to start at 0 and end at max yield + 2%
        ax.set_ylim(0, max([y * 100 for y in self.yields]) + 2)
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Yield (%)')
        ax.set_title('Yield Curve')
        ax.grid(True)
        return fig


class MarketSimulation:
    """A simulation of a financial market with various instruments"""
    def __init__(self):
        # Create bank bills with different maturities
        self.bank_bills = [
            BankBill(maturity_days=30, yield_rate=0.0436),
            BankBill(maturity_days=60, yield_rate=0.0435),
            BankBill(maturity_days=90, yield_rate=0.0436),
            BankBill(maturity_days=180, yield_rate=0.0439)
        ]

        # Create bonds with different maturities
        self.bonds = [
            Bond(maturity_years=1, coupon_rate=0.055, yield_to_maturity=0.0415),
            Bond(maturity_years=2, coupon_rate=0.057, yield_to_maturity=0.0400),
            Bond(maturity_years=5, coupon_rate=0.06, yield_to_maturity=0.0408),
            Bond(maturity_years=10, coupon_rate=0.065, yield_to_maturity=0.0451)
        ]

        # Create the yield curve
        self.yield_curve = YieldCurve(self.bank_bills, self.bonds)

        # Create FRAs
        self.fras = [
            ForwardRateAgreement(underlying_bill=self.bank_bills[2], settlement_days=90),
            ForwardRateAgreement(underlying_bill=self.bank_bills[2], settlement_days=180),
            ForwardRateAgreement(underlying_bill=self.bank_bills[3], settlement_days=90)
        ]

        # Create Bond Forwards
        self.bond_forwards = [
            BondForward(underlying_bond=self.bonds[0], settlement_days=90),
            BondForward(underlying_bond=self.bonds[1], settlement_days=180),
            BondForward(underlying_bond=self.bonds[2], settlement_days=90)
        ]
        
        # Initialize parameters for GBM model
        self.dt = 1/12  # Time step (representing one update)
        self.bill_drift = 0.001  # Annual drift for bank bill yields
        self.bond_drift = 0.002  # Annual drift for bond yields
        self.fra_drift = 0.0015  # Annual drift for FRA rates
        self.bond_forward_drift = 0.0025  # Annual drift for bond forward rates
        
        # Historical values for plotting
        self.history = {
            "bank_bills": {i: [] for i in range(len(self.bank_bills))},
            "bonds": {i: [] for i in range(len(self.bonds))},
            "fras": {i: [] for i in range(len(self.fras))},
            "bond_forwards": {i: [] for i in range(len(self.bond_forwards))}
        }
        self.record_history()

    def record_history(self):
        """Record current market values to history"""
        for i, bill in enumerate(self.bank_bills):
            self.history["bank_bills"][i].append(bill.yield_rate)
            
        for i, bond in enumerate(self.bonds):
            self.history["bonds"][i].append(bond.yield_to_maturity)
            
        for i, fra in enumerate(self.fras):
            self.history["fras"][i].append(fra.forward_rate)
            
        for i, bf in enumerate(self.bond_forwards):
            self.history["bond_forwards"][i].append(bf.forward_yield)

    def geometric_brownian_motion(self, current_value, drift, volatility, dt):
        """
        Simulate one step of Geometric Brownian Motion
        dS = μSdt + σSdW
        Where:
        - μ is the drift
        - σ is the volatility
        - dW is a Wiener process increment
        """
        # GBM formula: S(t+dt) = S(t) * exp((μ - σ²/2) * dt + σ * sqrt(dt) * Z)
        # Where Z is a standard normal random variable
        z = np.random.normal(0, 1)
        print(z)
        return current_value * np.exp((drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)

    def update_market(self, volatility: float = 0.05):
        """Update all market prices with Geometric Brownian Motion"""
        # Scale volatility by the user-defined factor
        vol_factor = volatility  # 0.1 to 1.0 range from UI
        
        # Update bank bills using GBM
        for bill in self.bank_bills:
            new_yield = self.geometric_brownian_motion(
                bill.yield_rate, 
                self.bill_drift, 
                0.12 * vol_factor,  # Base volatility scaled by user factor
                self.dt
            )
            bill.update_yield(max(0.001, new_yield))  # Ensure positive yield

        # Update bonds using GBM
        for bond in self.bonds:
            new_ytm = self.geometric_brownian_motion(
                bond.yield_to_maturity,
                self.bond_drift,
                0.12 * vol_factor,  # Higher volatility for longer-term instruments
                self.dt
            )
            bond.update_ytm(max(0.001, new_ytm))
            print(new_ytm, bond)
        
        print("market step")

        # Update yield curve
        self.yield_curve.update_curve()

        # Update FRAs with some deviation from theoretical prices using GBM
        for fra in self.fras:
            theoretical_rate = fra.calculate_theoretical_forward_rate()
            # Apply GBM around the theoretical rate
            deviation = self.geometric_brownian_motion(
                0.002,  # Base deviation amount
                self.fra_drift,
                0.15 * vol_factor,
                self.dt
            ) - 0.002  # Center around zero
            new_rate = theoretical_rate + deviation
            fra.update_forward_rate(max(0.001, new_rate))

        # Update Bond Forwards with some deviation from theoretical prices using GBM
        for bf in self.bond_forwards:
            theoretical_yield = bf.calculate_theoretical_forward_yield()
            # Apply GBM around the theoretical yield
            deviation = self.geometric_brownian_motion(
                0.003,  # Base deviation amount
                self.bond_forward_drift,
                0.18 * vol_factor,
                self.dt
            ) - 0.003  # Center around zero
            new_yield = theoretical_yield + deviation
            bf.update_forward_yield(max(0.001, new_yield))
            
        # Record new values to history
        self.record_history()

    def plot_rate_history(self, instrument_type):
        """Plot the history of rates for the given instrument type"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = self.history[instrument_type]
        labels = []
        
        for i, rates in data.items():
            if len(rates) > 1:  # Only plot if we have history
                if instrument_type == "bank_bills":
                    label = f"Bill {i+1} ({self.bank_bills[i].maturity_days} days)"
                elif instrument_type == "bonds":
                    label = f"Bond {i+1} ({self.bonds[i].maturity_years} years)"
                elif instrument_type == "fras":
                    label = f"FRA {i+1} ({self.fras[i].settlement_days} days)"
                else:  # bond_forwards
                    label = f"BF {i+1} ({self.bond_forwards[i].settlement_days} days)"
                
                ax.plot(rates, label=label)
                labels.append(label)
        
        ax.set_xlabel('Market Updates')
        
        if instrument_type in ["bank_bills", "bonds"]:
            ax.set_ylabel('Yield (%)')
            title = "Yield History"
        elif instrument_type == "fras":
            ax.set_ylabel('Forward Rate (%)')
            title = "Forward Rate History"
        else:
            ax.set_ylabel('Forward Yield (%)')
            title = "Forward Yield History"
            
        # Convert to percentage for display
        yticks = ax.get_yticks()
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{x*100:.2f}%' for x in yticks])
        
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        return fig

    def get_arbitrage_opportunities(self) -> Dict:
        """Get all arbitrage opportunities in the market"""
        opportunities = {
            "fra": [],
            "bond_forward": []
        }

        for i, fra in enumerate(self.fras):
            has_opp, diff = fra.calculate_arbitrage_opportunity()
            if has_opp:
                opportunities["fra"].append({
                    "instrument": f"FRA {i+1}",
                    "description": f"Settlement: {fra.settlement_days} days, Bill Maturity: {fra.underlying_bill.maturity_days} days",
                    "market_price": fra.price,
                    "theoretical_price": fra.calculate_price_from_forward_rate(fra.calculate_theoretical_forward_rate()),
                    "difference": diff,
                    "action": "Buy" if diff < 0 else "Sell"
                })

        for i, bf in enumerate(self.bond_forwards):
            has_opp, diff = bf.calculate_arbitrage_opportunity()
            if has_opp:
                opportunities["bond_forward"].append({
                    "instrument": f"Bond Forward {i+1}",
                    "description": f"Settlement: {bf.settlement_days} days, Bond Maturity: {bf.underlying_bond.maturity_years} years",
                    "market_price": bf.price,
                    "theoretical_price": bf.calculate_price_from_forward_yield(bf.calculate_theoretical_forward_yield()),
                    "difference": diff,
                    "action": "Buy" if diff < 0 else "Sell"
                })

        return opportunities

# ---------------------- Streamlit App ----------------------

def main():
    st.set_page_config(page_title="Financial Market Simulator", layout="wide")

    st.title("Financial Market Simulator")
    st.markdown("""
    This application simulates a financial market with various instruments:
    - Bank Bills (short-term debt instruments)
    - Bonds (longer-term debt instruments)
    - Forward Rate Agreements (FRAs)
    - Bond Forwards
    
    The simulation shows how prices move around and identifies arbitrage opportunities.
    """)

    # Initialize or get simulation from session state
    if 'market_sim' not in st.session_state:
        st.session_state.market_sim = MarketSimulation()
        st.session_state.volatility = 0.5
        st.session_state.update_count = 0

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Simulation Controls")

        volatility = st.slider("Market Volatility", 
                              min_value=0.1, 
                              max_value=1.0, 
                              value=st.session_state.volatility,
                              step=0.1)
        st.session_state.volatility = volatility

        if st.button("Update Market"):
            st.session_state.market_sim.update_market(volatility)
            st.session_state.update_count += 1

        st.text(f"Market Updates: {st.session_state.update_count}")

        auto_update = st.checkbox("Auto-update Market")

        if auto_update:
            time_interval = st.slider("Update Interval (seconds)", 1, 10, 3)
            st.markdown(f"Market will update every {time_interval} seconds")

            # This will automatically update the market data
            import time
            time.sleep(time_interval)
            st.session_state.market_sim.update_market(volatility)
            st.session_state.update_count += 1
            st.rerun()

    with col2:
        st.subheader("Yield Curve")
        st.pyplot(st.session_state.market_sim.yield_curve.plot())
    
    # Add tabs for historical charts
    history_tab1, history_tab2, history_tab3, history_tab4 = st.tabs(["Bank Bill History", "Bond History", "FRA History", "Bond Forward History"])
    
    with history_tab1:
        if st.session_state.update_count > 1:
            st.pyplot(st.session_state.market_sim.plot_rate_history("bank_bills"))
        else:
            st.info("Run market updates to see historical yield movements")
            
    with history_tab2:
        if st.session_state.update_count > 1:
            st.pyplot(st.session_state.market_sim.plot_rate_history("bonds"))
        else:
            st.info("Run market updates to see historical yield movements")
            
    with history_tab3:
        if st.session_state.update_count > 1:
            st.pyplot(st.session_state.market_sim.plot_rate_history("fras"))
        else:
            st.info("Run market updates to see historical forward rate movements")
            
    with history_tab4:
        if st.session_state.update_count > 1:
            st.pyplot(st.session_state.market_sim.plot_rate_history("bond_forwards"))
        else:
            st.info("Run market updates to see historical forward yield movements")

    st.header("Market Data")

    # Create tabs for different instrument types
    tab1, tab2, tab3, tab4 = st.tabs(["Bank Bills", "Bonds", "Forward Rate Agreements", "Bond Forwards"])

    with tab1:
        st.subheader("Bank Bills")

        # Create a DataFrame from bank bills for display
        bill_data = []
        for i, bill in enumerate(st.session_state.market_sim.bank_bills):
            bill_data.append({
                "Bank Bill": f"Bill {i+1}",
                "Maturity (days)": bill.maturity_days,
                "Price ($)": f"{bill.price:.2f}",
                "Yield (%)": f"{bill.yield_rate*100:.2f}%"
            })

        st.table(pd.DataFrame(bill_data))

    with tab2:
        st.subheader("Bonds")

        # Create a DataFrame from bonds for display
        bond_data = []
        for i, bond in enumerate(st.session_state.market_sim.bonds):
            bond_data.append({
                "Bond": f"Bond {i+1}",
                "Maturity (years)": bond.maturity_years,
                "Coupon (%)": f"{bond.coupon_rate*100:.2f}%",
                "Price ($)": f"{bond.price:.2f}",
                "YTM (%)": f"{bond.yield_to_maturity*100:.2f}%"
            })

        st.table(pd.DataFrame(bond_data))

    with tab3:
        st.subheader("Forward Rate Agreements (FRAs)")

        # Create a DataFrame from FRAs for display
        fra_data = []
        for i, fra in enumerate(st.session_state.market_sim.fras):
            has_opp, diff = fra.calculate_arbitrage_opportunity()

            fra_data.append({
                "FRA": f"FRA {i+1}",
                "Underlying Bill Maturity": f"{fra.underlying_bill.maturity_days} days",
                "Settlement (days)": fra.settlement_days,
                "Price ($)": f"{fra.price:.2f}",
                "Forward Rate (%)": f"{fra.forward_rate*100:.2f}%",
                "Theoretical Rate (%)": f"{fra.calculate_theoretical_forward_rate()*100:.2f}%",
                "Arbitrage": "YES" if has_opp else "NO",
                "Profit Potential": f"${abs(diff):.2f}"
            })

        st.table(pd.DataFrame(fra_data))

    with tab4:
        st.subheader("Bond Forwards")

        # Create a DataFrame from Bond Forwards for display
        bf_data = []
        for i, bf in enumerate(st.session_state.market_sim.bond_forwards):
            has_opp, diff = bf.calculate_arbitrage_opportunity()

            bf_data.append({
                "Bond Forward": f"BF {i+1}",
                "Underlying Bond Maturity": f"{bf.underlying_bond.maturity_years} years",
                "Settlement (days)": bf.settlement_days,
                "Price ($)": f"{bf.price:.2f}",
                "Forward Yield (%)": f"{bf.forward_yield*100:.2f}%",
                "Theoretical Yield (%)": f"{bf.calculate_theoretical_forward_yield()*100:.2f}%",
                "Arbitrage": "YES" if has_opp else "NO",
                "Profit Potential": f"${abs(diff):.2f}"
            })

        st.table(pd.DataFrame(bf_data))

    # Arbitrage Opportunities Section
    st.header("Arbitrage Opportunities")

    opportunities = st.session_state.market_sim.get_arbitrage_opportunities()

    if not opportunities["fra"] and not opportunities["bond_forward"]:
        st.info("No arbitrage opportunities currently exist in the market.")
    else:
        if opportunities["fra"]:
            st.subheader("FRA Arbitrage Opportunities")
            fra_opp_data = pd.DataFrame(opportunities["fra"])
            st.table(fra_opp_data)

        if opportunities["bond_forward"]:
            st.subheader("Bond Forward Arbitrage Opportunities")
            bf_opp_data = pd.DataFrame(opportunities["bond_forward"])
            st.table(bf_opp_data)

        st.markdown("""
        **Trading Strategy:**
        - **Buy** when market price is below theoretical price (undervalued)
        - **Sell** when market price is above theoretical price (overvalued)
        """)

if __name__ == "__main__":
    main()