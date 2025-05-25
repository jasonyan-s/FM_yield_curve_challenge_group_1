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
        temp_bill.set_ytm(yield_rate)
        print(f" face value is {self.face_value}, maturity is {self.maturity_days/365}, yield_rate is {yield_rate}, price is {temp_bill.price}")
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
        temp_bond.set_ytm(ytm)

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
        
        print(f"Estimated YTM from price ytm_low is ytm_high is ${price:.2f}: {ytm_mid*100:.2f}%")
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
        has_opportunity = abs(diff) > 10  # Threshold for meaningful arbitrage
        
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
        has_opportunity = abs(diff) > 20  # Threshold for meaningful arbitrage
        
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
        ax.set_ylim([min([y * 100 for y in self.yields]) - 2, max([y * 100 for y in self.yields]) + 2])
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
    

    def update_market(self, base_volatility=0.05, bill_vol_factor=1.0, bond_vol_factor=1.0, 
                 fra_vol_factor=1.2, bf_vol_factor=1.3, drift=0.03,
                 short_medium_corr=0.7, medium_long_corr=0.5):
        """Update all market prices using correlated Geometric Brownian Motion"""
        # Set time step (dt) - assuming each update represents 1 day in market time
        dt = 1/252  # Standard assumption: ~252 trading days per year
        
        # Apply drift parameter 
        mu = drift  # Annual drift from parameter
        
        # Scale volatility parameter to match GBM expectations
        sigma_base = base_volatility * 0.1  # Scale input volatility to reasonable range
        
        # Generate correlated random numbers for the yield curve
        # We'll use 3 correlated random numbers for short, medium, long rates
        correlation_matrix = np.array([
            [1.0, short_medium_corr, short_medium_corr * medium_long_corr],
            [short_medium_corr, 1.0, medium_long_corr],
            [short_medium_corr * medium_long_corr, medium_long_corr, 1.0]
        ])
        
        # Cholesky decomposition for generating correlated random numbers
        try:
            L = np.linalg.cholesky(correlation_matrix)
            z = np.random.normal(0, 1, 3)  # Standard normal random variables
            correlated_randoms = np.dot(L, z)  # Correlated random variables
        except np.linalg.LinAlgError:  # If cholesky fails, use uncorrelated
            correlated_randoms = np.random.normal(0, 1, 3)
        
        # Update bank bills based on maturity
        for i, bill in enumerate(self.bank_bills):
            # Determine which part of the curve this belongs to
            maturity_years = bill.maturity_days / 365
            
            # Select appropriate random number based on maturity
            if maturity_years <= 0.5:
                diffusion = sigma_base * bill_vol_factor * correlated_randoms[0] * np.sqrt(dt)
            elif maturity_years <= 2:
                diffusion = sigma_base * bill_vol_factor * correlated_randoms[1] * np.sqrt(dt)
            else:
                diffusion = sigma_base * bill_vol_factor * correlated_randoms[2] * np.sqrt(dt)
        
            # Apply GBM to the yield
            drift_term = mu * dt
            
            # Yield follows GBM but can't go below a minimum threshold
            yield_change_factor = np.exp(drift_term + diffusion)
            new_yield = max(0.001, bill.yield_rate * yield_change_factor)
            bill.update_yield(new_yield)
        
        # Update bonds based on maturity
        for bond in self.bonds:
            # Determine which part of the curve this belongs to
            if bond.maturity_years <= 2:
                diffusion = sigma_base * bond_vol_factor * correlated_randoms[1] * np.sqrt(dt)
            else:
                diffusion = sigma_base * bond_vol_factor * correlated_randoms[2] * np.sqrt(dt)
            
            # Apply GBM to the yield-to-maturity
            drift_term = mu * dt
            
            # YTM follows GBM but can't go below a minimum threshold
            ytm_change_factor = np.exp(drift_term + diffusion)
            new_ytm = max(0.001, bond.yield_to_maturity * ytm_change_factor)
            bond.update_ytm(new_ytm)
        
        # Update yield curve
        self.yield_curve.update_curve()
        
        # Update FRAs with some deviation from theoretical prices using GBM
        for fra in self.fras:
            theoretical_rate = fra.calculate_theoretical_forward_rate()
            
            # Determine volatility factor based on maturity
            maturity_years = fra.underlying_bill.maturity_days / 365
            if maturity_years <= 0.5:
                random_idx = 0
            elif maturity_years <= 2:
                random_idx = 1
            else:
                random_idx = 2
                
            # Apply GBM to add realistic noise to the theoretical rate
            drift_term = mu * dt
            diffusion = sigma_base * fra_vol_factor * correlated_randoms[random_idx] * np.sqrt(dt)
            
            rate_change_factor = np.exp(drift_term + diffusion)
            new_rate = max(0.001, theoretical_rate * rate_change_factor)
            fra.update_forward_rate(new_rate)
        
        # Update Bond Forwards with some deviation from theoretical prices using GBM
        for bf in self.bond_forwards:
            theoretical_yield = bf.calculate_theoretical_forward_yield()
            
            # Determine volatility factor based on maturity
            if bf.underlying_bond.maturity_years <= 2:
                random_idx = 1
            else:
                random_idx = 2
                
            # Apply GBM to add realistic noise to the theoretical yield
            drift_term = mu * dt
            diffusion = sigma_base * bf_vol_factor * correlated_randoms[random_idx] * np.sqrt(dt)
            
            yield_change_factor = np.exp(drift_term + diffusion)
            new_yield = max(0.001, theoretical_yield * yield_change_factor)
            bf.update_forward_yield(new_yield)
        
        # Update the portfolio
        self.create_portfolio()
    
    
    def get_arbitrage_opportunities(self) -> Dict:
        """Get all arbitrage opportunities in the market"""
        opportunities = {
            "bank_bill": [],
            "bond": [],
            "fra": [],
            "bond_forward": []
        }
        
        # Check for arbitrage in FRAs (existing functionality)
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
        
        # Check for arbitrage in Bond Forwards (existing functionality)
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
        
        # NEW: Check for yield curve arbitrage between bank bills
        # Compare each bank bill with the yield curve's interpolated rate
        for i, bill in enumerate(self.bank_bills):
            maturity_years = bill.maturity_days / 365
            interpolated_rate = self.yield_curve.get_interpolated_rate(maturity_years)
            print(f"Interpolated rate for {bill.maturity_days} days: {interpolated_rate*100:.2f}%")
            
            # Calculate theoretical price based on interpolated rate
            theoretical_price = bill.calculate_price_from_yield(interpolated_rate)
            print(f"Theoretical price for {bill.maturity_days} days: ${theoretical_price:.2f}")
            print(f"Bill market price: ${bill.price:.2f}")
            diff = bill.price - theoretical_price
            
            # If difference is significant, consider it an arbitrage opportunity
            if abs(diff) > 10:  # Threshold for meaningful arbitrage
                opportunities["bank_bill"].append({
                    "instrument": f"Bank Bill {i+1}",
                    "description": f"Maturity: {bill.maturity_days} days",
                    "market_price": bill.price,
                    "theoretical_price": theoretical_price,
                    "difference": diff,
                    "action": "Buy" if diff < 0 else "Sell",
                    "market_rate": f"{bill.yield_rate*100:.2f}%",
                    "curve_rate": f"{interpolated_rate*100:.2f}%"
                })
        
        # NEW: Check for yield curve arbitrage between bonds
        # Compare each bond with the yield curve's interpolated rate
        for i, bond in enumerate(self.bonds):
            interpolated_rate = self.yield_curve.get_interpolated_rate(bond.maturity_years)
            
            # Calculate theoretical price based on interpolated rate
            theoretical_price = bond.calculate_price_from_ytm(interpolated_rate)
            diff = bond.price - theoretical_price
            
            # If difference is significant, consider it an arbitrage opportunity
            if abs(diff) > 20:  # Threshold for meaningful arbitrage
                opportunities["bond"].append({
                    "instrument": f"Bond {i+1}",
                    "description": f"Maturity: {bond.maturity_years} years, Coupon: {bond.coupon_rate*100:.2f}%",
                    "market_price": bond.price,
                    "theoretical_price": theoretical_price,
                    "difference": diff,
                    "action": "Buy" if diff < 0 else "Sell",
                    "market_rate": f"{bond.yield_to_maturity*100:.2f}%",
                    "curve_rate": f"{interpolated_rate*100:.2f}%"
                })
        
        print(opportunities)
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

    # Create sidebar for simulation parameters
    st.sidebar.header("Simulation Parameters")
    
    # Yield curve parameters
    st.sidebar.subheader("Yield Curve Parameters")
    rate_30d = st.sidebar.slider("30-day Rate (%)", 1.0, 10.0, 4.5, 0.1) / 100
    rate_60d = st.sidebar.slider("60-day Rate (%)", 1.0, 10.0, 4.7, 0.1) / 100
    rate_90d = st.sidebar.slider("90-day Rate (%)", 1.0, 10.0, 5.0, 0.1) / 100
    rate_180d = st.sidebar.slider("180-day Rate (%)", 1.0, 10.0, 5.3, 0.1) / 100
    rate_1y = st.sidebar.slider("1-year Rate (%)", 1.0, 10.0, 5.6, 0.1) / 100
    rate_2y = st.sidebar.slider("2-year Rate (%)", 1.0, 10.0, 5.8, 0.1) / 100
    rate_5y = st.sidebar.slider("5-year Rate (%)", 1.0, 10.0, 6.2, 0.1) / 100
    rate_10y = st.sidebar.slider("10-year Rate (%)", 1.0, 10.0, 6.7, 0.1) / 100
    
    # Volatility parameters
    st.sidebar.subheader("Volatility Parameters")
    bill_volatility = st.sidebar.slider("Bank Bill Volatility", 0.1, 1.0, 0.5, 0.1)
    bond_volatility = st.sidebar.slider("Bond Volatility", 0.1, 1.0, 0.5, 0.1)
    fra_volatility = st.sidebar.slider("FRA Volatility", 0.1, 1.5, 0.7, 0.1)
    bond_forward_volatility = st.sidebar.slider("Bond Forward Volatility", 0.1, 1.5, 0.8, 0.1)
    
    # Correlation parameters
    st.sidebar.subheader("Correlation Parameters")
    short_medium_correlation = st.sidebar.slider("Short-Medium Correlation", -1.0, 1.0, 0.7, 0.1)
    medium_long_correlation = st.sidebar.slider("Medium-Long Correlation", -1.0, 1.0, 0.6, 0.1)
    
    # Market update behavior
    st.sidebar.subheader("Market Update Behavior")
    market_drift = st.sidebar.slider("Market Drift (%/year)", -5.0, 5.0, 3.0, 0.1) / 100
    
    # Initialize or update simulation
    if 'market_sim' not in st.session_state or st.sidebar.button("Reset Simulation"):
        with st.spinner("Initializing market simulation..."):
            # Create custom market simulation with user parameters
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
            st.session_state.volatility = bill_volatility  # Default volatility
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
            # Initialize cumulative arbitrage tracking with all instrument types
            st.session_state.arbitrage_history = {
                "bank_bill": [],
                "bond": [],
                "fra": [],
                "bond_forward": []
            }
    
    # Main content
    st.markdown("""
    This application simulates a financial market with various instruments:
    - Bank Bills (short-term debt instruments)
    - Bonds (longer-term debt instruments)
    - Forward Rate Agreements (FRAs)
    - Bond Forwards
    
    The simulation shows real-time price movements and identifies arbitrage opportunities.
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
                                bill_vol_factor=bill_volatility,
                                bond_vol_factor=bond_volatility,
                                fra_vol_factor=fra_volatility,
                                bf_vol_factor=bond_forward_volatility,
                                drift=market_drift,
                                short_medium_corr=short_medium_correlation,
                                medium_long_corr=medium_long_correlation
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
                        initial_short_rate = rate_30d
                        initial_medium_rate = rate_2y
                        initial_long_rate = rate_10y
                        
                        # Reset rates to initial values
                        for i, bill in enumerate(st.session_state.market_sim.bank_bills):
                            maturity_years = bill.maturity_days / 365
                            if maturity_years <= 0.5:
                                bill.update_yield(initial_short_rate)
                            elif maturity_years <= 2:
                                bill.update_yield(initial_medium_rate)
                            else:
                                bill.update_yield(initial_long_rate)
                        
                        for i, bond in enumerate(st.session_state.market_sim.bonds):
                            if bond.maturity_years <= 2:
                                bond.update_ytm(initial_medium_rate)
                            else:
                                bond.update_ytm(initial_long_rate)
                                
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
                
                max_yield = max(max([y * 100 for y in first_yields] or [0]), max([y * 100 for y in last_yields] or [0]))
                min_yield = min(min([y * 100 for y in first_yields] or [0]), min([y * 100 for y in last_yields] or [0]))
                ax.set_ylim([min_yield - 0.5, max_yield + 0.5])
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

                                print(temp_bond)
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
        arb_tab1, arb_tab2, arb_tab3 = st.tabs(["All Opportunities", "FRA Opportunities", "Bond Forward Opportunities"])
        
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
            st.subheader("Bank Bill Arbitrage Opportunities")
            
            # Prepare Bank Bill opportunities for display
            bank_bill_opps = []
            for opp in st.session_state.arbitrage_history["bank_bill"]:
                bank_bill_opps.append({
                    "Update": opp["update_count"],
                    "Time": opp["timestamp"],
                    "Instrument": opp["instrument"],
                    "Description": opp["description"],
                    "Market Price": f"${opp['market_price']:.2f}",
                    "Theoretical Price": f"${opp['theoretical_price']:.2f}",
                    "Difference": f"${abs(opp['difference']):.2f}",
                    "Market Rate": opp["market_rate"],
                    "Curve Rate": opp["curve_rate"],
                    "Action": opp["action"],
                })
            
            # Sort by update count (most recent first)
            bank_bill_opps = sorted(bank_bill_opps, key=lambda x: x["Update"], reverse=True)
            
            # Display as dataframe
            if bank_bill_opps:
                st.dataframe(
                    bank_bill_opps,
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
                st.info("No Bank Bill arbitrage opportunities detected so far.")
        
        with arb_tab3:
            st.subheader("Bond Arbitrage Opportunities")
            
            # Prepare Bond opportunities for display
            bond_opps = []
            for opp in st.session_state.arbitrage_history["bond"]:
                bond_opps.append({
                    "Update": opp["update_count"],
                    "Time": opp["timestamp"],
                    "Instrument": opp["instrument"],
                    "Description": opp["description"],
                    "Market Price": f"${opp['market_price']:.2f}",
                    "Theoretical Price": f"${opp['theoretical_price']:.2f}",
                    "Difference": f"${abs(opp['difference']):.2f}",
                    "Market Rate": opp["market_rate"],
                    "Curve Rate": opp["curve_rate"],
                    "Action": opp["action"],
                })
            
            # Sort by update count (most recent first)
            bond_opps = sorted(bond_opps, key=lambda x: x["Update"], reverse=True)
            
            # Display as dataframe
            if bond_opps:
                st.dataframe(
                    bond_opps,
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
                st.info("No Bond arbitrage opportunities detected so far.")
        
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
                bill_vol_factor=bill_volatility,
                bond_vol_factor=bond_volatility,
                fra_vol_factor=fra_volatility,
                bf_vol_factor=bond_forward_volatility,
                drift=market_drift,
                short_medium_corr=short_medium_correlation,
                medium_long_corr=medium_long_correlation
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

def create_custom_market_simulation(
    rate_30d=0.045, rate_60d=0.047, rate_90d=0.05, rate_180d=0.053,
    rate_1y=0.056, rate_2y=0.058, rate_5y=0.062, rate_10y=0.067):
    """Create a market simulation with specified yield rates for standard tenors"""
    
    # Create bank bills with specific maturities and rates
    bank_bills = [
        BankBill(maturity_days=30, yield_rate=rate_30d),
        BankBill(maturity_days=60, yield_rate=rate_60d),
        BankBill(maturity_days=90, yield_rate=rate_90d),
        BankBill(maturity_days=180, yield_rate=rate_180d)
    ]
    
    # Create bonds with specific maturities and rates
    bonds = [
        Bond(maturity_years=1, coupon_rate=rate_1y - 0.002, yield_to_maturity=rate_1y),
        Bond(maturity_years=2, coupon_rate=rate_2y - 0.002, yield_to_maturity=rate_2y),
        Bond(maturity_years=5, coupon_rate=rate_5y - 0.002, yield_to_maturity=rate_5y),
        Bond(maturity_years=10, coupon_rate=rate_10y - 0.002, yield_to_maturity=rate_10y)
    ]
    
    # Create market simulation
    market_sim = MarketSimulation()
    
    # Replace default instruments with our custom ones
    market_sim.bank_bills = bank_bills
    market_sim.bonds = bonds
    
    # Create the yield curve
    market_sim.yield_curve = YieldCurve(market_sim.bank_bills, market_sim.bonds)
    
    # Create standard FRAs
    market_sim.fras = [
        ForwardRateAgreement(underlying_bill=market_sim.bank_bills[2], settlement_days=90),  # 90-day bill, 90-day settlement
        ForwardRateAgreement(underlying_bill=market_sim.bank_bills[2], settlement_days=180), # 90-day bill, 180-day settlement
        ForwardRateAgreement(underlying_bill=market_sim.bank_bills[3], settlement_days=90)   # 180-day bill, 90-day settlement
    ]
    
    # Create standard Bond Forwards
    market_sim.bond_forwards = [
        BondForward(underlying_bond=market_sim.bonds[0], settlement_days=90),  # 1-year bond, 90-day settlement
        BondForward(underlying_bond=market_sim.bonds[1], settlement_days=180), # 2-year bond, 180-day settlement
        BondForward(underlying_bond=market_sim.bonds[2], settlement_days=90)   # 5-year bond, 90-day settlement
    ]
    
    # Create a portfolio of all instruments
    market_sim.create_portfolio()
    
    return market_sim

if __name__ == "__main__":
    main()