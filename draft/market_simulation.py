from typing import List, Dict
import numpy as np
import curve_classes_and_functions
import instrument_classes
# market_simulation.py
import matplotlib.pyplot as plt

# Import or define the instrument classes used throughout this file
from instruments import BankBill, Bond
from derivatives import ForwardRateAgreement, BondForward


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
        ax.set_ylim([min([y * 100 for y in self.yields]) - 0.5, max([y * 100 for y in self.yields]) + 0.5])
        ax.set_ylabel('Yield (%)')
        ax.set_title('Yield Curve')
        ax.grid(True)
        for x, y in zip(self.maturities, self.yields):
            ax.annotate(f"{y*100:.2f}%\n({x:.2f}y)", (x, y*100), textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='blue')
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
            "bond_forward": [],
            "multi_instrument": []  # New category for multi-instrument opportunities
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
            
            # Calculate theoretical price based on interpolated rate
            theoretical_price = bill.calculate_price_from_yield(interpolated_rate)

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
        
        # Add multi-instrument opportunities
        multi_opps = self.get_multi_instrument_arbitrage()
        opportunities["multi_instrument"] = (
            multi_opps["butterfly"] + 
            multi_opps["calendar_spread"]
        )
        
        return opportunities

    def get_butterfly_arbitrage(self) -> List[Dict]:
        """Detect butterfly arbitrage opportunities in yield curve"""
        opportunities = []
        
        # Check bank bill butterfly opportunities
        for i in range(len(self.bank_bills) - 2):
            short_bill = self.bank_bills[i]
            mid_bill = self.bank_bills[i + 1]
            long_bill = self.bank_bills[i + 2]
            
            # Calculate theoretical middle rate based on linear interpolation
            mid_days = mid_bill.maturity_days
            weight = (mid_days - short_bill.maturity_days) / (long_bill.maturity_days - short_bill.maturity_days)
            theoretical_mid_yield = short_bill.yield_rate + weight * (long_bill.yield_rate - short_bill.yield_rate)
            
            # If actual middle rate deviates significantly from theoretical, it's an arbitrage opportunity
            if abs(mid_bill.yield_rate - theoretical_mid_yield) > 0.0005:  # 5 basis points threshold
                opportunities.append({
                    "type": "Butterfly",
                    "instruments": [
                        f"Bank Bill {i+1} ({short_bill.maturity_days}d)",
                        f"Bank Bill {i+2} ({mid_bill.maturity_days}d)",
                        f"Bank Bill {i+3} ({long_bill.maturity_days}d)"
                    ],
                    "action": "Buy Wings, Sell Body" if mid_bill.yield_rate > theoretical_mid_yield else "Sell Wings, Buy Body",
                    "expected_profit": abs(mid_bill.yield_rate - theoretical_mid_yield) * 10000,  # Convert to basis points
                    "details": f"Middle rate: {mid_bill.yield_rate*100:.3f}% vs Theoretical: {theoretical_mid_yield*100:.3f}%"
                })
        
        # Similar check for bonds
        for i in range(len(self.bonds) - 2):
            short_bond = self.bonds[i]
            mid_bond = self.bonds[i + 1]
            long_bond = self.bonds[i + 2]
            
            # Calculate theoretical middle yield based on linear interpolation
            mid_years = mid_bond.maturity_years
            weight = (mid_years - short_bond.maturity_years) / (long_bond.maturity_years - short_bond.maturity_years)
            theoretical_mid_ytm = short_bond.yield_to_maturity + weight * (long_bond.yield_to_maturity - short_bond.yield_to_maturity)
            
            if abs(mid_bond.yield_to_maturity - theoretical_mid_ytm) > 0.0005:
                opportunities.append({
                    "type": "Butterfly",
                    "instruments": [
                        f"Bond {i+1} ({short_bond.maturity_years}y)",
                        f"Bond {i+2} ({mid_bond.maturity_years}y)",
                        f"Bond {i+3} ({long_bond.maturity_years}y)"
                    ],
                    "action": "Buy Wings, Sell Body" if mid_bond.yield_to_maturity > theoretical_mid_ytm else "Sell Wings, Buy Body",
                    "expected_profit": abs(mid_bond.yield_to_maturity - theoretical_mid_ytm) * 10000,
                    "details": f"Middle YTM: {mid_bond.yield_to_maturity*100:.3f}% vs Theoretical: {theoretical_mid_ytm*100:.3f}%"
                })
                
        return opportunities
    
    def get_calendar_spread_arbitrage(self) -> List[Dict]:
        """Detect calendar spread arbitrage opportunities in forwards"""
        opportunities = []
        
        # Check for FRA calendar spread opportunities
        for i in range(len(self.fras)):
            for j in range(i + 1, len(self.fras)):
                fra1 = self.fras[i]
                fra2 = self.fras[j]
                
                # Only compare FRAs with same underlying but different settlement
                if fra1.underlying_bill.maturity_days == fra2.underlying_bill.maturity_days:
                    theoretical_spread = (fra2.calculate_theoretical_forward_rate() - 
                                       fra1.calculate_theoretical_forward_rate())
                    actual_spread = fra2.forward_rate - fra1.forward_rate
                    
                    if abs(actual_spread - theoretical_spread) > 0.0005:
                        opportunities.append({
                            "type": "Calendar Spread",
                            "instruments": [
                                f"FRA {i+1} (Settle: {fra1.settlement_days}d)",
                                f"FRA {j+1} (Settle: {fra2.settlement_days}d)"
                            ],
                            "action": ("Buy Near/Sell Far" if actual_spread > theoretical_spread 
                                     else "Sell Near/Buy Far"),
                            "expected_profit": abs(actual_spread - theoretical_spread) * 10000,
                            "details": f"Actual spread: {actual_spread*100:.3f}% vs Theoretical: {theoretical_spread*100:.3f}%"
                        })
        
        # Similar check for bond forwards
        for i in range(len(self.bond_forwards)):
            for j in range(i + 1, len(self.bond_forwards)):
                bf1 = self.bond_forwards[i]
                bf2 = self.bond_forwards[j]
                
                if bf1.underlying_bond.maturity_years == bf2.underlying_bond.maturity_years:
                    theoretical_spread = (bf2.calculate_theoretical_forward_yield() - 
                                       bf1.calculate_theoretical_forward_yield())
                    actual_spread = bf2.forward_yield - bf1.forward_yield
                    
                    if abs(actual_spread - theoretical_spread) > 0.0005:
                        opportunities.append({
                            "type": "Calendar Spread",
                            "instruments": [
                                f"Bond Forward {i+1} (Settle: {bf1.settlement_days}d)",
                                f"Bond Forward {j+1} (Settle: {bf2.settlement_days}d)"
                            ],
                            "action": ("Buy Near/Sell Far" if actual_spread > theoretical_spread 
                                     else "Sell Near/Buy Far"),
                            "expected_profit": abs(actual_spread - theoretical_spread) * 10000,
                            "details": f"Actual spread: {actual_spread*100:.3f}% vs Theoretical: {theoretical_spread*100:.3f}%"
                        })
        
        return opportunities

    def get_triangulation_arbitrage(self) -> List[Dict]:
        """Detect triangulation arbitrage opportunities between instruments"""
        opportunities = []
        
        # Check for triangulation between bank bill, FRA, and implied forward
        for bill in self.bank_bills:
            for fra in self.fras:
                if fra.underlying_bill.maturity_days == bill.maturity_days:
                    # Calculate implied forward rate from spot rates
                    spot_rate = bill.yield_rate
                    forward_rate = fra.forward_rate
                    settlement_years = fra.settlement_days / 365
                    maturity_years = bill.maturity_days / 365
                    
                    # Calculate theoretical forward rate from spot rates
                    implied_forward = ((1 + spot_rate * (settlement_years + maturity_years)) / 
                                    (1 + spot_rate * settlement_years) - 1) * (365 / bill.maturity_days)
                    
                    # If difference between actual and implied forward rate is significant
                    if abs(forward_rate - implied_forward) > 0.0005:  # 5 basis points threshold
                        opportunities.append({
                            "type": "Triangulation",
                            "instruments": [
                                f"Bank Bill ({bill.maturity_days}d)",
                                f"FRA (Settle: {fra.settlement_days}d)",
                                "Implied Forward"
                            ],
                            "action": ("Buy Bill+FRA/Sell Forward" if forward_rate > implied_forward 
                                     else "Sell Bill+FRA/Buy Forward"),
                            "expected_profit": abs(forward_rate - implied_forward) * 10000,  # Convert to basis points
                            "details": (f"Forward Rate: {forward_rate*100:.3f}% vs "
                                      f"Implied: {implied_forward*100:.3f}%")
                        })
        
        # Check for triangulation between bond, bond forward, and implied forward
        for bond in self.bonds:
            for bf in self.bond_forwards:
                if bf.underlying_bond.maturity_years == bond.maturity_years:
                    # Calculate implied forward yield from spot rates
                    spot_yield = bond.yield_to_maturity
                    forward_yield = bf.forward_yield
                    settlement_years = bf.settlement_days / 365
                    
                    # Calculate theoretical forward yield using spot-forward relationship
                    implied_forward_yield = ((1 + spot_yield) ** (bond.maturity_years) / 
                                          (1 + spot_yield) ** (settlement_years) - 1) / (
                                              bond.maturity_years - settlement_years)
                    
                    # If difference between actual and implied forward yield is significant
                    if abs(forward_yield - implied_forward_yield) > 0.0005:
                        opportunities.append({
                            "type": "Triangulation",
                            "instruments": [
                                f"Bond ({bond.maturity_years}y)",
                                f"Bond Forward (Settle: {bf.settlement_days}d)",
                                "Implied Forward"
                            ],
                            "action": ("Buy Bond+Forward/Sell Implied" if forward_yield > implied_forward_yield 
                                     else "Sell Bond+Forward/Buy Implied"),
                            "expected_profit": abs(forward_yield - implied_forward_yield) * 10000,
                            "details": (f"Forward Yield: {forward_yield*100:.3f}% vs "
                                      f"Implied: {implied_forward_yield*100:.3f}%")
                        })
        
        return opportunities

    def get_multi_instrument_arbitrage(self) -> Dict[str, List[Dict]]:
        """Get all multi-instrument arbitrage opportunities"""
        return {
            "butterfly": self.get_butterfly_arbitrage(),
            "calendar_spread": self.get_calendar_spread_arbitrage(),
            "triangulation": self.get_triangulation_arbitrage()
        }

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