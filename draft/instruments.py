# instruments.py
import instrument_classes
from dataclasses import dataclass
from typing import Optional

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

