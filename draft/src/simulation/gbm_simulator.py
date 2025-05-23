import numpy as np

class GBMSimulator:
    """
    Simulates interest rate movements using Geometric Brownian Motion.
    """
    
    def __init__(self, initial_rate, mu, sigma, dt=1/252):
        """
        Initialize the GBM simulator.
        
        Parameters:
        - initial_rate: Initial interest rate value
        - mu: Drift parameter (annualized)
        - sigma: Volatility parameter (annualized)
        - dt: Time step size (default: daily = 1/252)
        """
        self.current_rate = initial_rate
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.history = [initial_rate]
        
    def next_step(self):
        """
        Generate the next interest rate value using GBM.
        
        Returns:
            float: The next simulated rate
        """
        # Generate random normal increment
        z = np.random.normal(0, 1)
        
        # Apply GBM formula
        self.current_rate *= np.exp((self.mu - 0.5 * self.sigma**2) * self.dt + 
                                    self.sigma * np.sqrt(self.dt) * z)
        
        # Store in history
        self.history.append(self.current_rate)
        
        return self.current_rate
    
    def simulate_path(self, n_steps):
        """
        Simulate multiple steps at once.
        
        Parameters:
            n_steps (int): Number of steps to simulate
            
        Returns:
            list: Simulated rates
        """
        rates = []
        for _ in range(n_steps):
            rates.append(self.next_step())
        return rates
    
    def reset(self, initial_rate=None):
        """
        Reset the simulation to the initial state or a new initial rate.
        
        Parameters:
            initial_rate (float, optional): New initial rate
        """
        if initial_rate is not None:
            self.current_rate = initial_rate
            self.history = [initial_rate]
        else:
            self.current_rate = self.history[0]
            self.history = [self.history[0]]