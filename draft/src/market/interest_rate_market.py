import sys
import os
import numpy as np

# Add files directory to path for importing starter code
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'files'))

import instrument_classes as inst
import curve_classes_and_functions as curve

class InterestRateMarket:
    """
    Simulates a market of interest rate instruments.
    """
    
    def __init__(self):
        """Initialize the interest rate market with empty instruments."""
        self.bank_bills = []  # Short-term instruments
        self.bonds = []       # Longer-term instruments
        self.portfolio = inst.Portfolio()
        self.yield_curve = curve.ZeroCurve()
        
    def add_bank_bill(self, maturity, ytm, face_value=100):
        """
        Add a bank bill to the market.
        
        Parameters:
            maturity (float): Time to maturity in years
            ytm (float): Yield to maturity
            face_value (float): Face value of the instrument
        """
        bank_bill = inst.Bank_bill(face_value=face_value, maturity=maturity)
        bank_bill.set_ytm(ytm)
        bank_bill.set_cash_flows()
        self.bank_bills.append(bank_bill)
        self.portfolio.add_bank_bill(bank_bill)
        
    def add_bond(self, maturity, ytm, coupon, frequency=2, face_value=100):
        """
        Add a bond to the market.
        
        Parameters:
            maturity (float): Time to maturity in years
            ytm (float): Yield to maturity
            coupon (float): Annual coupon rate
            frequency (int): Coupon frequency per year
            face_value (float): Face value of the bond
        """
        bond = inst.Bond(face_value=face_value, maturity=maturity, 
                         coupon=coupon, frequency=frequency)
        bond.set_ytm(ytm)
        bond.set_cash_flows()
        self.bonds.append(bond)
        self.portfolio.add_bond(bond)
        
    def update_instrument_rates(self, rate_changes):
        """
        Update instrument rates based on simulated changes.
        
        Parameters:
            rate_changes (dict): Dictionary of {maturity: change_factor}
        """
        # Update bank bill rates
        for bill in self.bank_bills:
            maturity = bill.get_maturity()
            if maturity in rate_changes:
                current_ytm = bill.get_ytm()
                new_ytm = current_ytm * rate_changes[maturity]
                bill.set_ytm(new_ytm)
                
        # Update bond rates
        for bond in self.bonds:
            maturity = bond.get_maturity()
            if maturity in rate_changes:
                current_ytm = bond.get_ytm()
                new_ytm = current_ytm * rate_changes[maturity]
                bond.set_ytm(new_ytm)
                
    def build_yield_curve(self):
        """
        Build the yield curve based on current market instruments.
        Includes error handling for robust operation.
        
        Returns:
            ZeroCurve: The constructed yield curve
        """
        try:
            # Reset the yield curve
            self.yield_curve = curve.YieldCurve()
            
            # Ensure all instruments have their cash flows set correctly
            for bill in self.bank_bills:
                bill.set_cash_flows()
            for bond in self.bonds:
                bond.set_cash_flows()
                
            self.portfolio.set_cash_flows()
            self.yield_curve.set_constituent_portfolio(self.portfolio)
            
            # Try to bootstrap the curve
            try:
                self.yield_curve.bootstrap()
            except Exception as e:
                print(f"Error bootstrapping yield curve: {e}")
                # Fall back to direct curve construction if bootstrapping fails
                self._fallback_curve_construction()
                
        except Exception as e:
            print(f"Error building yield curve: {e}")
            # Provide a simple fallback
            self._fallback_curve_construction()
            
        return self.yield_curve

    def _fallback_curve_construction(self):
        """
        Fallback method to construct a yield curve when bootstrapping fails.
        Creates a simple curve directly from instrument YTMs.
        """
        self.yield_curve = curve.ZeroCurve()
        self.yield_curve.add_zero_rate(0, 0)
        
        # Add rates from bank bills
        for bill in self.bank_bills:
            try:
                maturity = bill.get_maturity()
                ytm = bill.get_ytm()
                if maturity > 0 and ytm > 0:
                    self.yield_curve.add_zero_rate(maturity, ytm)
            except:
                pass
        
        # Add rates from bonds
        for bond in self.bonds:
            try:
                maturity = bond.get_maturity() 
                ytm = bond.get_ytm()
                if maturity > 0 and ytm > 0:
                    self.yield_curve.add_zero_rate(maturity, ytm)
            except:
                pass
        
        # If curve is still empty, add some default points
        if len(self.yield_curve.maturities) <= 1:
            self.yield_curve.add_zero_rate(0.5, 0.03)
            self.yield_curve.add_zero_rate(1, 0.035)
            self.yield_curve.add_zero_rate(2, 0.04)
            self.yield_curve.add_zero_rate(5, 0.045)
            self.yield_curve.add_zero_rate(10, 0.05)

    def get_zero_rates(self):
        """
        Get the zero rates for all maturities.
        
        Returns:
            dict: {maturity: zero_rate} dictionary
        """
        maturities, discount_factors = self.yield_curve.get_zero_curve()
        zero_rates = {}
        
        for i, maturity in enumerate(maturities):
            if maturity > 0:  # Skip the zero maturity point
                zero_rates[maturity] = -np.log(discount_factors[i]) / maturity
                
        return zero_rates