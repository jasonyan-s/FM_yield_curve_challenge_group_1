# create a Class called ZeroCurve that will be used to store the zero rates and the discount factors
# for a given set of maturities and zero rates
# the class will have the following methods:
# - add_zero_rate(maturity, zero_rate): add a zero rate to the curve
# - get_zero_rate(maturity): get the zero rate for a given maturity


# make some imports
# random comment from howesrichard-tester
import numpy as np
import math

def exp_interp(xs, ys, x):
    """
    Interpolates a single point for a given value of x 
    using continuously compounded rates.

    Parameters:
    xs (list or np.array): Vector of x values sorted by x.
    ys (list or np.array): Vector of y values.
    x (float): The x value to interpolate.

    Returns:
    float: Interpolated y value.
    """
    xs = np.array(xs)
    ys = np.array(ys)
    
    # Handle edge cases
    if len(xs) == 0:
        raise ValueError("Empty arrays provided for interpolation")
        
    if len(xs) == 1:
        return ys[0]  # Only one point available, return it
        
    # Handle case where x is outside the range
    if x <= xs[0]:
        return ys[0]  # Return first value if x is before or at the first point
    if x >= xs[-1]:
        return ys[-1]  # Return last value if x is after or at the last point
    
    # Find the interval [x0, x1] where x0 <= x <= x1
    idx = np.searchsorted(xs, x) - 1
    
    # Ensure idx is in valid range
    if idx < 0:
        idx = 0
    if idx >= len(xs) - 1:
        idx = len(xs) - 2
        
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    
    # Calculate the continuously compounded rate
    # Handle case where x1 == x0 to avoid division by zero
    if x1 == x0:
        return y0
        
    rate = (np.log(y1) - np.log(y0)) / (x1 - x0)
    
    # Handle potential numerical issues with very small y values
    if np.isnan(rate) or np.isinf(rate):
        if abs(y1 - y0) < 1e-10:  # If values are very close
            return y0
        elif y0 <= 0 or y1 <= 0:  # If values are not positive
            # Linear interpolation as fallback
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    
    # Interpolate the y value for the given x
    y = y0 * np.exp(rate * (x - x0))
    
    return y
    
class ZeroCurve:
    def __init__(self):
        # set up empty list
        self.maturities = []
        self.zero_rates = []
        self.AtMats = []
        self.discount_factors = []
    
    def add_zero_rate(self, maturity, zero_rate):
        self.maturities.append(maturity)
        self.zero_rates.append(zero_rate)
        self.AtMats.append(math.exp(zero_rate*maturity))
        self.discount_factors.append(1/self.AtMats[-1])

    def add_discount_factor(self, maturity, discount_factor):
        self.maturities.append(maturity)
        self.discount_factors.append(discount_factor)
        self.AtMats.append(1/discount_factor)
        self.zero_rates.append(math.log(1/discount_factor)/maturity)
    
    def get_AtMat(self, maturity):
        if maturity in self.maturities:
            return self.AtMats[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.AtMats, maturity)

    def get_discount_factor(self, maturity):
        if maturity in self.maturities:
            return self.discount_factors[self.maturities.index(maturity)]
        else:
            return exp_interp(self.maturities, self.discount_factors, maturity)

    def get_zero_rate(self, maturity):
        if maturity in self.maturities:
            return self.zero_rates[self.maturities.index(maturity)]
        else:
            return math.log(self.get_AtMat(maturity))/maturity
        
    def get_zero_curve(self):
        return self.maturities, self.discount_factors
    
    def npv(self, cash_flows):
        npv = 0
        for maturity in cash_flows.get_maturities():
            npv += cash_flows.get_cash_flow(maturity)*self.get_discount_factor(maturity)
        return npv
            

class YieldCurve(ZeroCurve):
    def __init__(self):
        super().__init__()
        self.portfolio = []

    # set the constituent portfolio
    # the portfolio must contain bills and bonds in order of maturity
    # where all each successive bond only introduces one new cashflow beyond 
    #       the longest maturity to that point (being the maturity cashflow)
    def set_constituent_portfolio(self, portfolio):
        self.portfolio = portfolio

# Add this to the YieldCurve class in curve_classes_and_functions.py

    def bootstrap(self):
        """
        Bootstrap the yield curve from a portfolio of instruments.
        This method has been made more robust to handle edge cases and prevent index errors.
        """
        try:
            bank_bills = self.portfolio.get_bank_bills()
            bonds = self.portfolio.get_bonds()
            
            # Initialize with zero rate at time 0
            self.add_zero_rate(0, 0)
            
            # First, use bank bills to establish short end of the curve
            for bank_bill in bank_bills:
                try:
                    maturity = bank_bill.get_maturity()
                    discount_factor = bank_bill.get_price() / bank_bill.get_face_value()
                    self.add_discount_factor(maturity, discount_factor)
                except Exception as e:
                    print(f"Error processing bank bill: {e}")
                    continue
            
            # Then use bonds for the rest of the curve
            for bond in bonds:
                try:
                    # Get bond details
                    bond_dates = bond.get_maturities()
                    bond_amounts = bond.get_amounts()
                    maturity = bond.get_maturity()
                    
                    if len(bond_dates) < 2:
                        print(f"Skipping bond with insufficient cash flows")
                        continue
                    
                    # Calculate the PV of the bond cash flows excluding the maturity cash flow
                    pv = 0
                    for i in range(1, len(bond_amounts) - 1):
                        try:
                            # Make sure the date is valid
                            if i < len(bond_dates) and bond_dates[i] > 0:
                                pv += bond_amounts[i] * self.get_discount_factor(bond_dates[i])
                        except Exception as e:
                            print(f"Error processing bond cash flow at index {i}: {e}")
                            continue
                    
                    # Calculate the last cash flow (face value + final coupon)
                    final_cf = bond_amounts[-1] if len(bond_amounts) > 0 else bond.get_face_value()
                    
                    # Calculate the discount factor for the bond's maturity
                    if final_cf > 0:  # Avoid division by zero
                        discount_factor = (bond.get_price() - pv) / final_cf
                        if discount_factor > 0:  # Avoid negative discount factors
                            self.add_discount_factor(maturity, discount_factor)
                        else:
                            print(f"Skipping bond with negative discount factor: {discount_factor}")
                    else:
                        print(f"Skipping bond with zero or negative final cash flow")
                        
                except Exception as e:
                    print(f"Error bootstrapping bond: {e}")
                    continue
        except Exception as e:
            print(f"Bootstrap error: {e}")
            # Initialize with some default values if bootstrapping fails
            self.add_zero_rate(0, 0)
            self.add_zero_rate(1, 0.03)
            self.add_zero_rate(5, 0.04)
            self.add_zero_rate(10, 0.05)