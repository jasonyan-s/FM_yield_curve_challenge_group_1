import sys
import os
import unittest
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from simulation.gbm_simulator import GBMSimulator

class TestGBMSimulator(unittest.TestCase):
    
    def setUp(self):
        self.initial_rate = 0.05
        self.mu = 0.001
        self.sigma = 0.02
        self.simulator = GBMSimulator(self.initial_rate, self.mu, self.sigma)
    
    def test_initialization(self):
        """Test if simulator initializes correctly."""
        self.assertEqual(self.simulator.current_rate, self.initial_rate)
        self.assertEqual(self.simulator.history, [self.initial_rate])
        self.assertEqual(self.simulator.mu, self.mu)
        self.assertEqual(self.simulator.sigma, self.sigma)
    
    def test_next_step(self):
        """Test if next_step generates a new rate."""
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Get the next rate
        next_rate = self.simulator.next_step()
        
        # Check that it's not the same as the initial rate
        self.assertNotEqual(next_rate, self.initial_rate)
        
        # Check that the history has been updated
        self.assertEqual(len(self.simulator.history), 2)
        self.assertEqual(self.simulator.history[-1], next_rate)
    
    def test_simulate_path(self):
        """Test if simulate_path generates multiple steps."""
        # Set a fixed seed for reproducibility
        np.random.seed(42)
        
        # Simulate 10 steps
        n_steps = 10
        path = self.simulator.simulate_path(n_steps)
        
        # Check that the path has the correct length
        self.assertEqual(len(path), n_steps)
        
        # Check that the history has been updated
        self.assertEqual(len(self.simulator.history), n_steps + 1)  # Initial + n_steps
    
    def test_reset(self):
        """Test if reset returns to initial state."""
        # First, simulate some steps
        _ = self.simulator.simulate_path(5)
        
        # Reset the simulator
        self.simulator.reset()
        
        # Check that the state has been reset
        self.assertEqual(self.simulator.current_rate, self.initial_rate)
        self.assertEqual(self.simulator.history, [self.initial_rate])
    
    def test_reset_with_new_initial(self):
        """Test reset with a new initial rate."""
        new_initial = 0.06
        self.simulator.reset(new_initial)
        
        self.assertEqual(self.simulator.current_rate, new_initial)
        self.assertEqual(self.simulator.history, [new_initial])

if __name__ == '__main__':
    unittest.main()