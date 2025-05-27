# derivatives.py
from instruments import BankBill, Bond
import instrument_classes
from dataclasses import dataclass
from typing import Tuple, List, Optional

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