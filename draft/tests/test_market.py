import sys
import os
import unittest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from market.interest_rate_market import InterestRateMarket

class TestInterestRateMarket(unittest.TestCase):
    
    def setUp(self):
        self.market = InterestRateMarket()
    
    def test_add_bank_bill(self):
        """Test adding a bank bill to the market."""
        maturity = 0.5
        ytm = 0.03
        face_value = 100
        
        self.market.add_bank_bill(maturity, ytm, face_value)
        
        self.assertEqual(len(self.market.bank_bills), 1)
        bill = self.market.bank_bills[0]
        
        self.assertEqual(bill.get_maturity(), maturity)
        self.assertEqual(bill.get_ytm(), ytm)
        self.assertEqual(bill.get_face_value(), face_value)
    
    def test_add_bond(self):
        """Test adding a bond to the market."""
        maturity = 3.0
        ytm = 0.05
        coupon = 0.04
        frequency = 2
        face_value = 100
        
        self.market.add_bond(maturity, ytm, coupon, frequency, face_value)
        
        self.assertEqual(len(self.market.bonds), 1)
        bond = self.market.bonds[0]
        
        self.assertEqual(bond.get_maturity(), maturity)
        self.assertEqual(bond.get_ytm(), ytm)
        self.assertEqual(bond.get_coupon_rate(), coupon)
        self.assertEqual(bond.get_face_value(), face_value)
    
    def test_update_instrument_rates(self):
        """Test updating instrument rates."""
        # Add instruments
        self.market.add_bank_bill(0.5, 0.03, 100)
        self.market.add_bond(2.0, 0.05, 0.04, 2, 100)
        
        # Define rate changes
        rate_changes = {0.5: 1.1, 2.0: 0.9}  # 10% increase for 0.5y, 10% decrease for 2y
        
        # Update rates
        self.market.update_instrument_rates(rate_changes)
        
        # Check updated rates
        self.assertAlmostEqual(self.market.bank_bills[0].get_ytm(), 0.033)  # 0.03 * 1.1
        self.assertAlmostEqual(self.market.bonds[0].get_ytm(), 0.045)    # 0.05 * 0.9

if __name__ == '__main__':
    unittest.main()