import sys
import os
import unittest

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from arbitrage.detector import ArbitrageDetector
from market.interest_rate_market import InterestRateMarket

# Import from files directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'files'))
import curve_classes_and_functions as curve

class TestArbitrageDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = ArbitrageDetector(threshold=0.001)
        
        # Create a simple zero curve
        self.zero_curve = curve.ZeroCurve()
        self.zero_curve.add_zero_rate(1, 0.05)
        self.zero_curve.add_zero_rate(2, 0.06)
        self.zero_curve.add_zero_rate(3, 0.07)
    
    def test_no_arbitrage(self):
        """Test with a normal curve (no arbitrage)."""
        # A normal curve has increasing rates and decreasing discount factors
        opportunities = self.detector.detect_curve_arbitrage(self.zero_curve)
        self.assertEqual(len(opportunities), 0)
    
    def test_arbitrage_detection(self):
        """Test with a curve containing arbitrage."""
        # Create a curve with an inversion (arbitrage opportunity)
        arb_curve = curve.ZeroCurve()
        arb_curve.add_zero_rate(1, 0.05)
        arb_curve.add_zero_rate(2, 0.06)
        # Add a rate that's lower than the previous one, creating an arbitrage opportunity
        arb_curve.add_zero_rate(3, 0.04)
        
        opportunities = self.detector.detect_curve_arbitrage(arb_curve)
        
        # Should detect one opportunity
        self.assertEqual(len(opportunities), 1)
        
        # Check the detected opportunity
        opportunity = opportunities[0]
        self.assertEqual(opportunity['maturity_short'], 2)
        self.assertEqual(opportunity['maturity_long'], 3)

if __name__ == '__main__':
    unittest.main()