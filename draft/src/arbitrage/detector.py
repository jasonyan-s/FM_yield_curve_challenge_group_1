import numpy as np
import pandas as pd

class ArbitrageDetector:
    """
    Detects arbitrage opportunities in the interest rate market.
    """
    
    def __init__(self, threshold=0.001):
        """
        Initialize the arbitrage detector.
        
        Parameters:
            threshold (float): Threshold for detecting arbitrage opportunities
        """
        self.threshold = threshold
    
    def detect_curve_arbitrage(self, yield_curve, maturities=None):
        """
        Detect arbitrage opportunities in the yield curve.
        
        Parameters:
            yield_curve: The yield curve object
            maturities (list, optional): Specific maturities to check
            
        Returns:
            list: List of arbitrage opportunities
        """
        curve_maturities, discount_factors = yield_curve.get_zero_curve()
        
        # If no specific maturities provided, use curve maturities
        if maturities is None:
            maturities = curve_maturities
            
        opportunities = []
        
        # Check for non-decreasing discount factors (arbitrage opportunity)
        for i in range(1, len(maturities)):
            if maturities[i] > maturities[i-1]:  # Ensure maturities are in order
                prev_df = yield_curve.get_discount_factor(maturities[i-1])
                curr_df = yield_curve.get_discount_factor(maturities[i])
                
                # In a normal curve, discount factors should decrease with maturity
                if curr_df > prev_df:
                    opportunity = {
                        'maturity_short': maturities[i-1],
                        'maturity_long': maturities[i],
                        'discount_short': prev_df,
                        'discount_long': curr_df,
                        'potential': (curr_df - prev_df) / prev_df
                    }
                    opportunities.append(opportunity)
                    
        return opportunities
    
    def check_cross_instrument_arbitrage(self, market):
        """
        Check for arbitrage opportunities between different instruments.
        
        Parameters:
            market: The interest rate market object
            
        Returns:
            list: List of arbitrage opportunities
        """
        opportunities = []
        
        # Get all bank bills and bonds
        bank_bills = market.bank_bills
        bonds = market.bonds
        
        # Compare YTMs of instruments with similar maturities
        for bill in bank_bills:
            bill_maturity = bill.get_maturity()
            bill_ytm = bill.get_ytm()
            
            for bond in bonds:
                bond_maturity = bond.get_maturity()
                bond_ytm = bond.get_ytm()
                
                # If maturities are close, check for significant YTM differences
                if abs(bill_maturity - bond_maturity) < 0.1:
                    ytm_diff = bond_ytm - bill_ytm
                    
                    if abs(ytm_diff) > self.threshold:
                        opportunity = {
                            'instrument1': f"Bank Bill {bill_maturity}y",
                            'instrument2': f"Bond {bond_maturity}y",
                            'ytm1': bill_ytm,
                            'ytm2': bond_ytm,
                            'difference': ytm_diff,
                            'potential': abs(ytm_diff) / min(bill_ytm, bond_ytm)
                        }
                        opportunities.append(opportunity)
                        
        return opportunities