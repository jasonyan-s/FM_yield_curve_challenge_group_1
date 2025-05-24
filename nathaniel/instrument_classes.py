import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class Instrument:
    """
    Base class for financial instruments.
    
    This class defines common attributes and methods for all financial instruments.
    """
    
    def __init__(self, face_value=100, maturity=1.0):
        """
        Initialize a financial instrument.
        
        Parameters:
        -----------
        face_value : float, optional
            Face value of the instrument (default: 100)
        maturity : float, optional
            Maturity of the instrument in years (default: 1.0)
        """
        self.face_value = face_value
        self.maturity = maturity
        self.ytm = None  # Yield to maturity
        self.cash_flows = []  # List of tuples (maturity, amount)
    
    def set_ytm(self, ytm):
        """
        Set the yield to maturity of the instrument.
        
        Parameters:
        -----------
        ytm : float
            Yield to maturity as a decimal
        """
        if ytm is None or ytm < 0:
            raise ValueError("YTM must be non-negative and not None.")
        self.ytm = ytm
    
    def set_cash_flows(self):
        """Set the cash flows of the instrument. To be implemented by subclasses."""
        pass
    
    def get_cash_flows(self):
        """
        Get the cash flows of the instrument.
        
        Returns:
        --------
        list: List of tuples (maturity, amount)
        """
        # Filter out zero or negative cash flows to avoid divide by zero in bootstrapping
        return [(m, a) for (m, a) in self.cash_flows if a > 0 and m > 0]
    
    def get_price(self, discount_rates=None):
        """
        Calculate the price of the instrument.
        
        Parameters:
        -----------
        discount_rates : callable, optional
            Function that returns discount rates for given maturities
            If None, uses internal YTM
        
        Returns:
        --------
        float: Price of the instrument
        """
        if not self.cash_flows:
            raise ValueError("Cash flows not set. Call set_cash_flows() first.")
        
        price = 0
        
        for maturity, amount in self.get_cash_flows():
            if discount_rates is None:
                # Use internal YTM if no discount rates provided
                if self.ytm is None:
                    raise ValueError("YTM not set. Call set_ytm() first.")
                discount_factor = 1 / ((1 + self.ytm) ** maturity)
            else:
                # Use provided discount rates
                rate = discount_rates(maturity)
                discount_factor = 1 / ((1 + rate) ** maturity)
            
            price += amount * discount_factor
        
        return price
    
    def get_duration(self):
        """
        Calculate the Macaulay duration of the instrument.
        
        Returns:
        --------
        float: Macaulay duration in years
        """
        if not self.cash_flows or self.ytm is None:
            raise ValueError("Cash flows or YTM not set.")
        
        price = self.get_price()
        duration = 0
        
        for maturity, amount in self.cash_flows:
            discount_factor = 1 / ((1 + self.ytm) ** maturity)
            duration += maturity * amount * discount_factor / price
        
        return duration
    
    def get_modified_duration(self):
        """
        Calculate the modified duration of the instrument.
        
        Returns:
        --------
        float: Modified duration
        """
        macaulay_duration = self.get_duration()
        return macaulay_duration / (1 + self.ytm)
    
    def get_convexity(self):
        """
        Calculate the convexity of the instrument.
        
        Returns:
        --------
        float: Convexity
        """
        if not self.cash_flows or self.ytm is None:
            raise ValueError("Cash flows or YTM not set.")
        
        price = self.get_price()
        convexity = 0
        
        for maturity, amount in self.cash_flows:
            discount_factor = 1 / ((1 + self.ytm) ** maturity)
            convexity += maturity * (maturity + 1) * amount * discount_factor / price / (1 + self.ytm) ** 2
        
        return convexity


class Bank_bill(Instrument):
    """
    A class to represent a bank bill (zero-coupon instrument).
    """
    
    def __init__(self, face_value=100, maturity=0.25):
        """
        Initialize a bank bill.
        
        Parameters:
        -----------
        face_value : float, optional
            Face value of the bill (default: 100)
        maturity : float, optional
            Maturity of the bill in years (default: 0.25)
        """
        super().__init__(face_value, maturity)
    
    def set_cash_flows(self):
        """Set the cash flows of the bank bill."""
        # Bank bill has only one cash flow at maturity
        if self.maturity <= 0 or self.face_value <= 0:
            raise ValueError("Bank bill must have positive maturity and face value.")
        self.cash_flows = [(self.maturity, self.face_value)]
    
    def get_price_from_ytm(self):
        """
        Calculate the price of the bank bill from YTM.
        
        Returns:
        --------
        float: Price of the bank bill
        """
        if self.ytm is None:
            raise ValueError("YTM not set. Call set_ytm() first.")
        
        return self.face_value / ((1 + self.ytm) ** self.maturity)
    
    def get_ytm_from_price(self, price):
        """
        Calculate the YTM of the bank bill from price.
        
        Parameters:
        -----------
        price : float
            Price of the bank bill
        
        Returns:
        --------
        float: Yield to maturity
        """
        return (self.face_value / price) ** (1 / self.maturity) - 1
    
    def get_maturity(self):
        """Get the maturity of the bank bill."""
        return self.maturity


class Bond(Instrument):
    """
    A class to represent a coupon-bearing bond.
    """
    
    def __init__(self, face_value=100, maturity=1, coupon=0.05, frequency=2):
        """
        Initialize a bond.
        
        Parameters:
        -----------
        face_value : float, optional
            Face value of the bond (default: 100)
        maturity : float, optional
            Maturity of the bond in years (default: 1)
        coupon : float, optional
            Annual coupon rate as a decimal (default: 0.05)
        frequency : int, optional
            Coupon payment frequency per year (default: 2)
        """
        super().__init__(face_value, maturity)
        self.coupon = coupon
        self.frequency = frequency
    
    def set_cash_flows(self):
        """Set the cash flows of the bond."""
        # Clear existing cash flows
        self.cash_flows = []
        if self.maturity <= 0 or self.face_value <= 0 or self.coupon < 0 or self.frequency <= 0:
            raise ValueError("Bond must have positive maturity, face value, frequency, and non-negative coupon.")
        period_length = 1 / self.frequency
        coupon_amount = self.face_value * self.coupon / self.frequency
        num_periods = int(self.maturity * self.frequency)
        for i in range(1, num_periods + 1):
            payment_time = i * period_length
            if i == num_periods:
                self.cash_flows.append((payment_time, coupon_amount + self.face_value))
            else:
                self.cash_flows.append((payment_time, coupon_amount))
    
    def get_ytm_from_price(self, price, max_iterations=100, tolerance=1e-10):
        """
        Calculate the YTM of the bond from price using Newton-Raphson method.
        
        Parameters:
        -----------
        price : float
            Price of the bond
        max_iterations : int, optional
            Maximum number of iterations (default: 100)
        tolerance : float, optional
            Convergence tolerance (default: 1e-10)
        
        Returns:
        --------
        float: Yield to maturity
        """
        if not self.cash_flows:
            raise ValueError("Cash flows not set. Call set_cash_flows() first.")
        
        # Initial guess for YTM (current coupon rate is a good starting point)
        ytm = self.coupon
        
        for _ in range(max_iterations):
            # Calculate price and derivative at current YTM estimate
            p = 0
            dp = 0
            
            for maturity, amount in self.cash_flows:
                df = 1 / ((1 + ytm) ** maturity)
                p += amount * df
                dp -= maturity * amount * df / (1 + ytm)
            
            # Check if current price is close enough to target
            if abs(p - price) < tolerance:
                break
            
            # Update YTM estimate using Newton-Raphson formula
            ytm = ytm - (p - price) / dp
        
        return ytm
    
    def get_maturity(self):
        """Get the maturity of the bond."""
        return self.maturity


class InflationLinkedBond(Bond):
    """
    A class to represent an inflation-linked bond.
    """
    
    def __init__(self, face_value=100, maturity=1, coupon=0.05, frequency=2, inflation_rate=0.02):
        """
        Initialize an inflation-linked bond.
        
        Parameters:
        -----------
        face_value : float, optional
            Face value of the bond (default: 100)
        maturity : float, optional
            Maturity of the bond in years (default: 1)
        coupon : float, optional
            Annual coupon rate as a decimal (default: 0.05)
        frequency : int, optional
            Coupon payment frequency per year (default: 2)
        inflation_rate : float, optional
            Annual inflation rate as a decimal (default: 0.02)
        """
        super().__init__(face_value, maturity, coupon, frequency)
        self.inflation_rate = inflation_rate
    
    def set_inflation_rate(self, inflation_rate):
        """
        Set the inflation rate.
        
        Parameters:
        -----------
        inflation_rate : float
            Annual inflation rate as a decimal
        """
        self.inflation_rate = inflation_rate
    
    def set_cash_flows(self):
        """Set the cash flows of the inflation-linked bond."""
        self.cash_flows = []
        if self.maturity <= 0 or self.face_value <= 0 or self.coupon < 0 or self.frequency <= 0:
            raise ValueError("Inflation-linked bond must have positive maturity, face value, frequency, and non-negative coupon.")
        period_length = 1 / self.frequency
        num_periods = int(self.maturity * self.frequency)
        for i in range(1, num_periods + 1):
            payment_time = i * period_length
            inflation_factor = (1 + self.inflation_rate) ** payment_time
            adjusted_coupon = self.face_value * self.coupon / self.frequency * inflation_factor
            if i == num_periods:
                adjusted_face_value = self.face_value * inflation_factor
                self.cash_flows.append((payment_time, adjusted_coupon + adjusted_face_value))
            else:
                self.cash_flows.append((payment_time, adjusted_coupon))
    
    def get_maturity(self):
        """Get the maturity of the inflation-linked bond."""
        return self.maturity


class Portfolio:
    """
    A class to represent a portfolio of financial instruments.
    """
    
    def __init__(self):
        """Initialize an empty portfolio."""
        self.bills = []
        self.bonds = []
        self.inflation_linked_bonds = []
        self.all_cash_flows = []
    
    def add_bank_bill(self, bill):
        """
        Add a bank bill to the portfolio.
        
        Parameters:
        -----------
        bill : Bank_bill
            Bank bill to add
        """
        self.bills.append(bill)
    
    def add_bond(self, bond):
        """
        Add a bond to the portfolio.
        
        Parameters:
        -----------
        bond : Bond
            Bond to add
        """
        self.bonds.append(bond)
    
    def add_inflation_linked_bond(self, bond):
        """
        Add an inflation-linked bond to the portfolio.
        
        Parameters:
        -----------
        bond : InflationLinkedBond
            Inflation-linked bond to add
        """
        self.inflation_linked_bonds.append(bond)
    
    def set_cash_flows(self):
        """Set the cash flows of all instruments in the portfolio."""
        self.all_cash_flows = []
        # Add cash flows from all instruments, filtering out zero/negative flows
        for bill in self.bills:
            self.all_cash_flows.extend(bill.get_cash_flows())
        for bond in self.bonds:
            self.all_cash_flows.extend(bond.get_cash_flows())
        for bond in self.inflation_linked_bonds:
            self.all_cash_flows.extend(bond.get_cash_flows())
        # Remove any zero or negative cash flows
        self.all_cash_flows = [(m, a) for (m, a) in self.all_cash_flows if a > 0 and m > 0]
    
    def get_all_cash_flows(self):
        """
        Get all cash flows in the portfolio.
        
        Returns:
        --------
        list: List of tuples (maturity, amount)
        """
        if not self.all_cash_flows:
            self.set_cash_flows()
        
        return self.all_cash_flows
    
    def get_value(self, discount_rates):
        """
        Calculate the total value of the portfolio.
        
        Parameters:
        -----------
        discount_rates : callable
            Function that returns discount rates for given maturities
        
        Returns:
        --------
        float: Total value of the portfolio
        """
        total_value = 0
        
        # Calculate value of each instrument
        for bill in self.bills:
            total_value += bill.get_price(discount_rates)
        
        for bond in self.bonds:
            total_value += bond.get_price(discount_rates)
        
        for bond in self.inflation_linked_bonds:
            total_value += bond.get_price(discount_rates)
        
        return total_value