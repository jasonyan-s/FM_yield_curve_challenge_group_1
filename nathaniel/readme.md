# Yield Curve Simulation Tool

This application simulates the evolution of yield curves over time using correlated geometric Brownian motion. It provides a powerful interface for visualizing and analyzing how yield curves might evolve under different market conditions.

## Project Structure

The project consists of the following files:

1. **yield_curve_app.py**: The main Streamlit application that provides the user interface.
2. **yield_curve_simulation.py**: Contains the `YieldCurveSimulator` class that handles the simulation logic.
3. **curve_classes_and_functions.py**: Defines the `YieldCurve` class and utility functions for curve operations.
4. **instrument_classes.py**: Contains classes for financial instruments (Bank bills, Bonds, etc.) used in the simulation.

## Features

- Simulate yield curves with control over parameters like:
  - Volatility for different parts of the curve
  - Drift parameters
  - Correlations between rates
  - Initial rate conditions
- Visualize:
  - Yield curve snapshots at specific time steps
  - Rate evolution over time for selected maturities
  - Access raw simulation data for further analysis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/yield-curve-simulation.git
cd yield-curve-simulation
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:

```bash
streamlit run yield_curve_app.py
```

### Simulation Parameters

Adjust the following parameters in the sidebar:

- **Timeframe**:
  - Time Horizon: Number of trading days to simulate
  - Number of Simulation Steps: Number of discrete steps in the simulation

- **Volatility Parameters**:
  - Volatility for short-term, medium-term, long-term rates, and inflation

- **Drift Parameters**:
  - Drift for short-term, medium-term, long-term rates, and inflation

- **Correlation Parameters**:
  - Correlations between each pair of rates

- **Initial Rate Parameters**:
  - Starting rates for the simulation

### Visualization

After running the simulation, you can explore the results through:

1. **Yield Curve Snapshots**: See how the shape of the yield curve changes at different time steps
2. **Rate Evolution**: Track specific maturities over time
3. **Data Table**: View and download the raw simulation data

## Technical Details

The simulation uses geometric Brownian motion with correlated random variables to model how rates evolve over time. The correlation structure allows for realistic co-movement of different parts of the yield curve.

### Methodology

1. The simulator generates correlated paths for short-term, medium-term, long-term rates, and inflation
2. At each step, financial instruments (bank bills and bonds) are created based on these rates
3. A yield curve is constructed using these instruments via bootstrapping
4. Zero rates are extracted for specific maturities

### Mathematical Model

The rates follow geometric Brownian motion:

```
dR_t = μR_t dt + σR_t dW_t
```

Where:
- R_t is the rate
- μ is the drift parameter
- σ is the volatility parameter
- dW_t is a Wiener process (Brownian motion)

Correlation between rates is handled using Cholesky decomposition of the correlation matrix.

## Requirements

- Python 3.7+
- streamlit
- numpy
- pandas
- matplotlib
- scipy

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.