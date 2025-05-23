import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation

class CurveVisualizer:
    """
    Visualizes yield curves and their changes over time.
    """
    
    def __init__(self, figsize=(10, 6)):
        """
        Initialize the visualizer.
        
        Parameters:
            figsize (tuple): Figure size for plots
        """
        self.figsize = figsize
        self.curve_history = []  # List of (maturities, rates) tuples
        self.fig, self.ax = None, None
        
    def add_curve(self, maturities, rates):
        """
        Add a curve to the history.
        
        Parameters:
            maturities (list): List of maturities
            rates (list): List of corresponding rates
        """
        self.curve_history.append((maturities, rates))
        
    def plot_current_curve(self, maturities, rates, title="Current Yield Curve"):
        """
        Plot the current yield curve.
        
        Parameters:
            maturities (list): List of maturities
            rates (list): List of corresponding rates
            title (str): Plot title
            
        Returns:
            (fig, ax): The figure and axis objects
        """
        # Filter out (0, 0) if present
        filtered = [(m, r) for m, r in zip(maturities, rates) if not (m == 0 and r == 0)]
        if filtered:
            maturities_f, rates_f = zip(*filtered)
        else:
            maturities_f, rates_f = [], []
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(maturities_f, rates_f, 'o-', linewidth=2, markersize=8)
        ax.set_title(title)
        ax.set_xlabel('Maturity (Years)')
        ax.set_ylabel('Zero Rate')
        ax.set_ylim(0, max(rates) + 0.005 if rates else 1)
        ax.grid(True)
        return fig, ax
        
    def create_animation(self, interval=200):
        """
        Create an animation of the yield curve over time.
        
        Parameters:
            interval (int): Time between frames in milliseconds
            
        Returns:
            ani: Animation object or None if no data
        """
        # Check if we have any curves to animate
        if not self.curve_history:
            print("No curve history to animate")
            return None
            
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.ax.set_xlabel('Maturity (Years)')
        self.ax.set_ylabel('Zero Rate')
        self.ax.set_title('Yield Curve Evolution')
        self.ax.grid(True)
        
        # Find the range of rates to set y-axis limits
        all_rates = []
        for _, rates in self.curve_history:
            all_rates.extend(rates)
        
        if all_rates:
            min_rate = min(all_rates) - 0.005
            max_rate = max(all_rates) + 0.005
            self.ax.set_ylim(min_rate, max_rate)
        
        line, = self.ax.plot([], [], 'o-', linewidth=2, markersize=8)
        
        def init():
            line.set_data([], [])
            return line,
        
        def animate(i):
            maturities, rates = self.curve_history[i]
            line.set_data(maturities, rates)
            self.ax.set_title(f'Yield Curve - Frame {i+1}/{len(self.curve_history)}')
            return line,
        
        ani = FuncAnimation(
            self.fig, animate, frames=len(self.curve_history),
            init_func=init, blit=True, interval=interval
        )
        
        return ani
    
    def plot_rate_history(self, maturity, rates_history, title=None):
        """
        Plot the history of a specific maturity's rate.
        
        Parameters:
            maturity (float): The maturity to plot
            rates_history (list): List of rates for this maturity
            title (str): Plot title
            
        Returns:
            (fig, ax): The figure and axis objects
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.plot(range(len(rates_history)), rates_history, 'b-', linewidth=2)
        ax.set_title(title or f'Rate History for {maturity}-Year Maturity')
        ax.set_xlabel('Simulation Step')
        ax.set_ylabel('Rate')
        ax.grid(True)
        return fig, ax