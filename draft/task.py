import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import instrument_classes
import curve_classes_and_functions
import time

# Set random seed for reproducibility
np.random.seed(42)

# ---------------------- Financial Instruments ----------------------

@dataclass
class BankBill:
    """A short-term debt instrument"""
    face_value: float = 1000000  # $1M face value
    maturity_days: int = 90  # 90 day bank bill
    price: float = None
    yield_rate: float = None
    
    def __post_init__(self):
        # Convert maturity from days to years for the underlying implementation
        maturity_years = self.maturity_days / 365
        
        # Create an instance of the teacher's Bank_bill class
        self.bill_impl = instrument_classes.Bank_bill(
            face_value=self.face_value,
            maturity=maturity_years,
            ytm=self.yield_rate if self.yield_rate is not None else None,
            price=self.price if self.price is not None else None
        )
        
        if self.price is None and self.yield_rate is None:
            # Default to 5% yield
            self.yield_rate = 0.05
            self.bill_impl.set_ytm(self.yield_rate)
            self.price = self.bill_impl.get_price()
        elif self.price is None:
            self.bill_impl.set_ytm(self.yield_rate)
            self.price = self.bill_impl.get_price()
        elif self.yield_rate is None:
            self.bill_impl.set_price(self.price)
            self.yield_rate = self.bill_impl.get_ytm()
            
        # Setup the cash flows
        self.bill_impl.set_cash_flows()
    
    def calculate_price_from_yield(self, yield_rate: float) -> float:
        """Calculate price from yield"""
        temp_bill = instrument_classes.Bank_bill(
            face_value=self.face_value,
            maturity=self.maturity_days/365,
            ytm=yield_rate
        )
        return temp_bill.get_price()
    
    def calculate_yield_from_price(self, price: float) -> float:
        """Calculate yield from price"""
        temp_bill = instrument_classes.Bank_bill(
            face_value=self.face_value,
            maturity=self.maturity_days/365,
            price=price
        )
        return temp_bill.get_ytm()
    
    def update_price(self, new_price: float):
        """Update price and recalculate yield"""
        self.price = new_price
        self.bill_impl.set_price(new_price)
        self.yield_rate = self.bill_impl.get_ytm()
        self.bill_impl.set_cash_flows()
    
    def update_yield(self, new_yield: float):
        """Update yield and recalculate price"""
        self.yield_rate = new_yield
        self.bill_impl.set_ytm(new_yield)
        self.price = self.bill_impl.get_price()
        self.bill_impl.set_cash_flows()
    
    def get_cash_flows(self):
        """Get the bank bill's cash flows"""
        return self.bill_impl.get_cash_flows()
    
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
        # Create an instance of the teacher's Bond class
        self.bond_impl = instrument_classes.Bond(
            face_value=self.face_value,
            maturity=self.maturity_years,
            coupon=self.coupon_rate,
            frequency=self.frequency,
            ytm=self.yield_to_maturity if self.yield_to_maturity is not None else self.coupon_rate,
            price=self.price if self.price is not None else None
        )
        
        # Initialize price and YTM based on the bond_impl calculations
        if self.price is None and self.yield_to_maturity is None:
            # Default to same YTM as coupon rate
            self.yield_to_maturity = self.coupon_rate
            self.bond_impl.set_ytm(self.yield_to_maturity)
            self.price = self.bond_impl.get_price()
        elif self.price is None:
            self.bond_impl.set_ytm(self.yield_to_maturity)
            self.price = self.bond_impl.get_price()
        elif self.yield_to_maturity is None:
            # Need to solve for YTM from price using our existing method
            self.yield_to_maturity = self.calculate_ytm_from_price(self.price)
            self.bond_impl.set_ytm(self.yield_to_maturity)
        
        # Setup the cash flows
        self.bond_impl.set_cash_flows()
    
    def calculate_price_from_ytm(self, ytm: float) -> float:
        """Calculate bond price from yield to maturity"""
        temp_bond = instrument_classes.Bond(
            face_value=self.face_value,
            maturity=self.maturity_years,
            coupon=self.coupon_rate,
            frequency=self.frequency,
            ytm=ytm
        )
        return temp_bond.get_price()
    
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
        self.bond_impl = instrument_classes.Bond(
            face_value=self.face_value,
            maturity=self.maturity_years,
            coupon=self.coupon_rate,
            frequency=self.frequency,
            ytm=self.yield_to_maturity,
            price=self.price
        )
        self.bond_impl.set_cash_flows()
    
    def update_ytm(self, new_ytm: float):
        """Update YTM and recalculate price"""
        self.yield_to_maturity = new_ytm
        self.bond_impl.set_ytm(new_ytm)
        self.price = self.bond_impl.get_price()
        self.bond_impl.set_cash_flows()
    
    def get_cash_flows(self):
        """Get the bond's cash flows"""
        return self.bond_impl.get_cash_flows()
    
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
        # Convert settlement days to years for calculations
        self.settlement_years = self.settlement_days / 365
        
        # Create cash flows
        self.cash_flows = instrument_classes.CashFlows()
        
        if self.price is None and self.forward_rate is None:
            # Calculate theoretical forward rate
            self.forward_rate = self.calculate_theoretical_forward_rate()
            self.price = self.calculate_price_from_forward_rate(self.forward_rate)
        elif self.price is None:
            self.price = self.calculate_price_from_forward_rate(self.forward_rate)
        elif self.forward_rate is None:
            self.forward_rate = self.calculate_forward_rate_from_price(self.price)
            
        # Set up cash flows
        self.set_cash_flows()
    
    def set_cash_flows(self):
        """Set up cash flows for the FRA"""
        self.cash_flows = instrument_classes.CashFlows()
        
        # At time 0, pay the price of the FRA
        self.cash_flows.add_cash_flow(0, -self.price)
        
        # At settlement, receive the bank bill (pay nothing)
        # Then at maturity of the bank bill, receive face value
        future_bill_maturity = self.settlement_years + (self.underlying_bill.maturity_days / 365)
        self.cash_flows.add_cash_flow(future_bill_maturity, self.underlying_bill.face_value)
    
    def calculate_theoretical_forward_rate(self) -> float:
        """Calculate the theoretical forward rate based on the yield curve"""
        # Calculate using the underlying bill's yield
        spot_rate = self.underlying_bill.yield_rate
        maturity = self.underlying_bill.maturity_days / 365
        settlement = self.settlement_days / 365
        
        # Forward rate formula based on no-arbitrage pricing
        numerator = (1 + spot_rate * (settlement + maturity))
        denominator = (1 + spot_rate * settlement)
        
        forward_rate = (numerator / denominator - 1) * (365 / self.underlying_bill.maturity_days)
        return forward_rate
    
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
        self.set_cash_flows()
    
    def update_forward_rate(self, new_rate: float):
        """Update forward rate and recalculate price"""
        self.forward_rate = new_rate
        self.price = self.calculate_price_from_forward_rate(new_rate)
        self.set_cash_flows()
    
    def get_cash_flows(self):
        """Get the FRA's cash flows"""
        return self.cash_flows.get_cash_flows()
    
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
        # Convert settlement days to years for calculations
        self.settlement_years = self.settlement_days / 365
        
        # Create cash flows
        self.cash_flows = instrument_classes.CashFlows()
        
        if self.price is None and self.forward_yield is None:
            # Calculate theoretical forward yield
            self.forward_yield = self.calculate_theoretical_forward_yield()
            self.price = self.calculate_price_from_forward_yield(self.forward_yield)
        elif self.price is None:
            self.price = self.calculate_price_from_forward_yield(self.forward_yield)
        elif self.forward_yield is None:
            self.forward_yield = self.calculate_forward_yield_from_price(self.price)
            
        # Set up cash flows
        self.set_cash_flows()
    
    def set_cash_flows(self):
        """Set up cash flows for the Bond Forward"""
        self.cash_flows = instrument_classes.CashFlows()
        
        # At time 0, pay the price of the forward
        self.cash_flows.add_cash_flow(0, -self.price)
        
        # Get the bond's cash flows and adjust them by the settlement time
        bond_cash_flows = self.underlying_bond.get_cash_flows()
        
        for t, amount in bond_cash_flows:
            # Skip the initial cash flow (bond price)
            if t == 0:
                continue
            # Add all other cash flows, shifted by settlement time
            self.cash_flows.add_cash_flow(self.settlement_years + t, amount)
    
    def calculate_theoretical_forward_yield(self) -> float:
        """Calculate the theoretical forward yield based on the yield curve"""
        # More sophisticated model using the underlying bond's YTM
        # and adjusting for the term structure
        current_yield = self.underlying_bond.yield_to_maturity
        
        # Simple model: forward yield increases slightly with settlement time
        forward_yield = current_yield + (0.002 * self.settlement_years)
        return forward_yield
    
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
        self.set_cash_flows()
    
    def update_forward_yield(self, new_yield: float):
        """Update forward yield and recalculate price"""
        self.forward_yield = new_yield
        self.price = self.calculate_price_from_forward_yield(new_yield)
        self.set_cash_flows()
    
    def get_cash_flows(self):
        """Get the Bond Forward's cash flows"""
        return self.cash_flows.get_cash_flows()
    
    def __str__(self) -> str:
        return f"BondForward(settlement={self.settlement_days} days, " \
               f"price=${self.price:.2f}, forward_yield={self.forward_yield*100:.2f}%)"


class YieldCurve:
    """A yield curve constructed from market instruments"""
    def __init__(self, bank_bills: List[BankBill], bonds: List[Bond]):
        # Create an underlying Portfolio from teacher's implementation
        self.portfolio = instrument_classes.Portfolio()
        
        self.bank_bills = sorted(bank_bills, key=lambda x: x.maturity_days)
        self.bonds = sorted(bonds, key=lambda x: x.maturity_years)
        
        # Add the instruments to the portfolio
        for bill in self.bank_bills:
            self.portfolio.add_bank_bill(bill.bill_impl)
        
        for bond in self.bonds:
            self.portfolio.add_bond(bond.bond_impl)
            
        # Create the yield curve implementation
        self.curve_impl = curve_classes_and_functions.YieldCurve()
        self.curve_impl.set_constituent_portfolio(self.portfolio)
        
        # Initialize maturities and yields lists
        self.maturities = []
        self.yields = []
        self.update_curve()
    
    def update_curve(self):
        """Update the yield curve points from current market instruments"""
        self.maturities = []
        self.yields = []
        
        # Update the portfolio with current instrument states
        self.portfolio = instrument_classes.Portfolio()
        for bill in self.bank_bills:
            self.portfolio.add_bank_bill(bill.bill_impl)
        
        for bond in self.bonds:
            self.portfolio.add_bond(bond.bond_impl)
        
        self.curve_impl.set_constituent_portfolio(self.portfolio)
        
        try:
            # Try to bootstrap the curve - this might not always work with arbitrary instruments
            self.curve_impl.bootstrap()
            
            # Get the curve points for visualization
            mats, rates = [], []
            
            # Add bank bill points
            for bill in self.bank_bills:
                mats.append(bill.maturity_days / 365)
                rates.append(bill.yield_rate)
            
            # Add bond points
            for bond in self.bonds:
                mats.append(bond.maturity_years)
                rates.append(bond.yield_to_maturity)
                
            self.maturities = mats
            self.yields = rates
        except Exception as e:
            # Fall back to simple curve construction if bootstrapping fails
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
            BankBill(maturity_days=30, yield_rate=0.045),
            BankBill(maturity_days=60, yield_rate=0.047),
            BankBill(maturity_days=90, yield_rate=0.05),
            BankBill(maturity_days=180, yield_rate=0.053)
        ]
        
        # Create bonds with different maturities
        self.bonds = [
            Bond(maturity_years=1, coupon_rate=0.055, yield_to_maturity=0.056),
            Bond(maturity_years=2, coupon_rate=0.057, yield_to_maturity=0.058),
            Bond(maturity_years=5, coupon_rate=0.06, yield_to_maturity=0.062),
            Bond(maturity_years=10, coupon_rate=0.065, yield_to_maturity=0.067)
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
        
        # Create a portfolio of all instruments
        self.create_portfolio()
    
    def create_portfolio(self):
        """Create a portfolio containing all market instruments"""
        self.portfolio = instrument_classes.Portfolio()
        
        # Add bank bills
        for bill in self.bank_bills:
            self.portfolio.add_bank_bill(bill.bill_impl)
        
        # Add bonds
        for bond in self.bonds:
            self.portfolio.add_bond(bond.bond_impl)
        
        # Set up cash flows for the portfolio
        self.portfolio.set_cash_flows()
        
        return self.portfolio
    
    def update_market(self, volatility: float = 0.05):
        """Update all market prices with random movements"""
        # Update bank bills
        for bill in self.bank_bills:
            change = np.random.normal(0, volatility) * 0.003  # Small random changes
            new_yield = max(0.001, bill.yield_rate + change)  # Ensure positive yield
            bill.update_yield(new_yield)
        
        # Update bonds
        for bond in self.bonds:
            change = np.random.normal(0, volatility) * 0.004
            new_ytm = max(0.001, bond.yield_to_maturity + change)
            bond.update_ytm(new_ytm)
        
        # Update yield curve
        self.yield_curve.update_curve()
        
        # Update FRAs with some deviation from theoretical prices
        for fra in self.fras:
            theoretical_rate = fra.calculate_theoretical_forward_rate()
            deviation = np.random.normal(0, volatility) * 0.006
            new_rate = theoretical_rate + deviation
            fra.update_forward_rate(max(0.001, new_rate))
        
        # Update Bond Forwards with some deviation from theoretical prices
        for bf in self.bond_forwards:
            theoretical_yield = bf.calculate_theoretical_forward_yield()
            deviation = np.random.normal(0, volatility) * 0.007
            new_yield = theoretical_yield + deviation
            bf.update_forward_yield(max(0.001, new_yield))
        
        # Update the portfolio
        self.create_portfolio()
    
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
    st.markdown("""
    This application simulates a financial market with various instruments:
    - Bank Bills (short-term debt instruments)
    - Bonds (longer-term debt instruments)
    - Forward Rate Agreements (FRAs)
    - Bond Forwards
    
    The simulation shows real-time price movements and identifies arbitrage opportunities.
    """)
    
    # Initialize or get simulation from session state
    if 'market_sim' not in st.session_state:
        st.session_state.market_sim = MarketSimulation()
        st.session_state.volatility = 0.5
        st.session_state.update_count = 0
        st.session_state.price_history = {
            'bank_bills': {i: [] for i in range(len(st.session_state.market_sim.bank_bills))},
            'bonds': {i: [] for i in range(len(st.session_state.market_sim.bonds))},
            'fras': {i: [] for i in range(len(st.session_state.market_sim.fras))},
            'bond_forwards': {i: [] for i in range(len(st.session_state.market_sim.bond_forwards))},
        }
        st.session_state.yield_history = []
        st.session_state.timestamps = []
        st.session_state.start_time = dt.datetime.now()
        # Initialize price change tracking
        st.session_state.previous_prices = {
            'bank_bills': [bill.price for bill in st.session_state.market_sim.bank_bills],
            'bonds': [bond.price for bond in st.session_state.market_sim.bonds],
            'fras': [fra.price for fra in st.session_state.market_sim.fras],
            'bond_forwards': [bf.price for bf in st.session_state.market_sim.bond_forwards],
        }
    
    # Create the layout
    left_col, right_col = st.columns([1, 3])
    
    with left_col:
        st.subheader("Market Controls")
        
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            volatility = st.slider("Market Volatility", 
                                  min_value=0.1, 
                                  max_value=1.0, 
                                  value=st.session_state.volatility,
                                  step=0.1,
                                  help="Higher volatility = larger price movements")
            st.session_state.volatility = volatility
            
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
                    
                    # Update the market
                    st.session_state.market_sim.update_market(volatility)
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
                    
                    # Add current yield curve snapshot
                    maturities = st.session_state.market_sim.yield_curve.maturities
                    yields = st.session_state.market_sim.yield_curve.yields
                    st.session_state.yield_history.append((maturities, yields))
            
            with col2:
                if st.button("Reset Simulation", use_container_width=True):
                    st.session_state.market_sim = MarketSimulation()
                    st.session_state.update_count = 0
                    st.session_state.price_history = {
                        'bank_bills': {i: [] for i in range(len(st.session_state.market_sim.bank_bills))},
                        'bonds': {i: [] for i in range(len(st.session_state.market_sim.bonds))},
                        'fras': {i: [] for i in range(len(st.session_state.market_sim.fras))},
                        'bond_forwards': {i: [] for i in range(len(st.session_state.market_sim.bond_forwards))},
                    }
                    st.session_state.yield_history = []
                    st.session_state.timestamps = []
                    st.session_state.start_time = dt.datetime.now()
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
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Market Summary Section
            st.subheader("Market Summary")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Display arbitrage opportunities summary
            opportunities = st.session_state.market_sim.get_arbitrage_opportunities()
            total_opportunities = len(opportunities["fra"]) + len(opportunities["bond_forward"])
            
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
                
            st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Yield Curve", "Price History"])
        
        with tab1:
            st.subheader("Dynamic Yield Curve")
            # Plot the current yield curve
            st.pyplot(st.session_state.market_sim.yield_curve.plot())
            
            # Add yield curve animation if we have history
            if len(st.session_state.yield_history) > 1:
                st.subheader("Yield Curve Evolution")
                # Create an animated plot of the yield curve over time
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot the first curve
                first_maturities, first_yields = st.session_state.yield_history[0]
                ax.plot(first_maturities, [y * 100 for y in first_yields], 'o-', alpha=0.3, color='blue')
                
                # Plot the latest curve
                last_maturities, last_yields = st.session_state.yield_history[-1]
                ax.plot(last_maturities, [y * 100 for y in last_yields], 'o-', linewidth=2, color='red')
                
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
                            ax.plot(range(len(history)), history, '-o', label=f"Bill {i+1} ({bill.maturity_days} days)")
                    
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
                            ax.plot(range(len(history)), history, '-o', label=f"Bond {i+1} ({bond.maturity_years} yrs)")
                    
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
                            ax.plot(range(len(history)), history, '-o', 
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
                            ax.plot(range(len(history)), history, '-o', 
                                   label=f"BF {i+1} (Bond: {bf.underlying_bond.maturity_years}y, Settle: {bf.settlement_days}d)")
                    
                    ax.set_xlabel('Market Updates')
                    ax.set_ylabel('Price ($)')
                    ax.set_title('Bond Forward Price History')
                    ax.grid(True)
                    ax.legend()
                    st.pyplot(fig)
            else:
                st.info("Run a few market updates to see price history charts")
    
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
    st.header("Arbitrage Opportunities Dashboard")
    
    opportunities = st.session_state.market_sim.get_arbitrage_opportunities()
    
    if not opportunities["fra"] and not opportunities["bond_forward"]:
        st.info("No arbitrage opportunities currently exist in the market.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if opportunities["fra"]:
                st.subheader("FRA Arbitrage Opportunities")
                
                for opp in opportunities["fra"]:
                    st.markdown(f"""
                    <div class="card arbitrage-opportunity">
                        <h4>{opp["instrument"]}</h4>
                        <p>{opp["description"]}</p>
                        <p>Market Price: <b>${opp["market_price"]:.2f}</b></p>
                        <p>Theoretical Price: <b>${opp["theoretical_price"]:.2f}</b></p>
                        <p>Difference: <b>${abs(opp["difference"]):.2f}</b></p>
                        <p>Action: <b style="color: {'green' if opp["action"] == 'Buy' else 'red'}">
                            {opp["action"]}
                        </b></p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            if opportunities["bond_forward"]:
                st.subheader("Bond Forward Arbitrage Opportunities")
                
                for opp in opportunities["bond_forward"]:
                    st.markdown(f"""
                    <div class="card arbitrage-opportunity">
                        <h4>{opp["instrument"]}</h4>
                        <p>{opp["description"]}</p>
                        <p>Market Price: <b>${opp["market_price"]:.2f}</b></p>
                        <p>Theoretical Price: <b>${opp["theoretical_price"]:.2f}</b></p>
                        <p>Difference: <b>${abs(opp["difference"]):.2f}</b></p>
                        <p>Action: <b style="color: {'green' if opp["action"] == 'Buy' else 'red'}">
                            {opp["action"]}
                        </b></p>
                    </div>
                    """, unsafe_allow_html=True)
        
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
        
        # Update the market
        st.session_state.market_sim.update_market(volatility)
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
        
        # Add current yield curve snapshot
        maturities = st.session_state.market_sim.yield_curve.maturities
        yields = st.session_state.market_sim.yield_curve.yields
        st.session_state.yield_history.append((maturities, yields))
        
        st.experimental_rerun()

if __name__ == "__main__":
    main()
