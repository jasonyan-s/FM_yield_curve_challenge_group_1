"""
Configuration parameters for the interest rate market simulation.
"""

# Simulation parameters
SIMULATION_STEPS = 100  # Number of steps to simulate
TIME_STEP = 1/252       # Daily steps (252 trading days per year)

# Market instruments configuration
BANK_BILLS = [
    # (maturity, initial_ytm, face_value)
    (0.25, 0.03, 100),  # 3-month bill
    (0.5, 0.035, 100),  # 6-month bill
    (0.75, 0.04, 100),  # 9-month bill
    (1.0, 0.045, 100),  # 1-year bill
]

BONDS = [
    # (maturity, initial_ytm, coupon, frequency, face_value)
    (2.0, 0.05, 0.045, 2, 100),  # 2-year bond
    (3.0, 0.055, 0.05, 2, 100),  # 3-year bond
    (5.0, 0.06, 0.055, 2, 100),  # 5-year bond
    (10.0, 0.065, 0.06, 2, 100), # 10-year bond
]

# GBM parameters for rate simulation
# Different maturities can have different parameters
GBM_PARAMETERS = {
    # maturity: (mu, sigma)
    0.25: (0.001, 0.03),  # Short-term rates: low drift, moderate volatility
    0.5: (0.001, 0.029),
    0.75: (0.001, 0.028),
    1.0: (0.001, 0.027),
    2.0: (0.0008, 0.025), # Medium-term rates: lower drift, lower volatility
    3.0: (0.0008, 0.023),
    5.0: (0.0005, 0.02),  # Long-term rates: lowest drift, lowest volatility
    10.0: (0.0003, 0.018),
}

# Correlation matrix between rates of different maturities
# Higher correlation between adjacent maturities
CORRELATION_MATRIX = {
    0.25: {0.25: 1.0, 0.5: 0.95, 0.75: 0.9, 1.0: 0.85, 2.0: 0.7, 3.0: 0.6, 5.0: 0.5, 10.0: 0.4},
    0.5: {0.25: 0.95, 0.5: 1.0, 0.75: 0.95, 1.0: 0.9, 2.0: 0.75, 3.0: 0.65, 5.0: 0.55, 10.0: 0.45},
    0.75: {0.25: 0.9, 0.5: 0.95, 0.75: 1.0, 1.0: 0.95, 2.0: 0.8, 3.0: 0.7, 5.0: 0.6, 10.0: 0.5},
    1.0: {0.25: 0.85, 0.5: 0.9, 0.75: 0.95, 1.0: 1.0, 2.0: 0.85, 3.0: 0.75, 5.0: 0.65, 10.0: 0.55},
    2.0: {0.25: 0.7, 0.5: 0.75, 0.75: 0.8, 1.0: 0.85, 2.0: 1.0, 3.0: 0.9, 5.0: 0.8, 10.0: 0.7},
    3.0: {0.25: 0.6, 0.5: 0.65, 0.75: 0.7, 1.0: 0.75, 2.0: 0.9, 3.0: 1.0, 5.0: 0.9, 10.0: 0.8},
    5.0: {0.25: 0.5, 0.5: 0.55, 0.75: 0.6, 1.0: 0.65, 2.0: 0.8, 3.0: 0.9, 5.0: 1.0, 10.0: 0.9},
    10.0: {0.25: 0.4, 0.5: 0.45, 0.75: 0.5, 1.0: 0.55, 2.0: 0.7, 3.0: 0.8, 5.0: 0.9, 10.0: 1.0},
}

# Visualization parameters
VISUALIZATION = {
    'figsize': (12, 8),
    'animation_interval': 200,  # milliseconds between frames
    'curve_color': 'blue',
    'arbitrage_color': 'red',
}