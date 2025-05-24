import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime, timedelta

class YieldCurve:
    """
    A class to represent a yield curve constructed from financial instruments.
    Implements robust bootstrapping for zero-coupon and coupon bonds.
    """

    def __init__(self):
        self.maturities = []
        self.zero_rates = []
        self.discount_factors = []
        self.constituents = None
        self.interp_method = 'linear'
        self.interpolator = None

    def set_constituent_portfolio(self, portfolio):
        self.constituents = portfolio

    def bootstrap(self):
        """
        Robust bootstrapping for a portfolio of zero-coupon and coupon bonds.
        Uses actual instrument prices (from get_price()), not just par.
        """
        if self.constituents is None:
            raise ValueError("No constituent instruments set. Use set_constituent_portfolio() first.")

        all_cash_flows = self.constituents.get_all_cash_flows()
        if not all_cash_flows:
            raise ValueError("No cash flows found in constituent instruments.")

        # Group cash flows by maturity
        cf_by_mat = {}
        for mat, amt in all_cash_flows:
            if mat not in cf_by_mat:
                cf_by_mat[mat] = []
            cf_by_mat[mat].append(amt)
        unique_maturities = sorted(cf_by_mat.keys())

        # Build a list of all instruments for price lookup
        all_instruments = []
        if hasattr(self.constituents, 'bills'):
            all_instruments.extend(self.constituents.bills)
        if hasattr(self.constituents, 'bonds'):
            all_instruments.extend(self.constituents.bonds)
        if hasattr(self.constituents, 'inflation_linked_bonds'):
            all_instruments.extend(self.constituents.inflation_linked_bonds)

        self.maturities = []
        self.zero_rates = []
        self.discount_factors = []

        prev_maturities = []
        prev_dfs = []

        for mat in unique_maturities:
            # Find all instruments that have a cash flow at this maturity
            candidate_instruments = []
            for inst in all_instruments:
                cfs = inst.get_cash_flows()
                if cfs and any(abs(cf[0] - mat) < 1e-8 for cf in cfs):
                    candidate_instruments.append(inst)

            # Use the price of the instrument that has its final cash flow at this maturity (i.e., mat is the last cash flow)
            inst_price = None
            for inst in candidate_instruments:
                cfs = inst.get_cash_flows()
                if abs(cfs[-1][0] - mat) < 1e-8:
                    try:
                        inst_price = inst.get_price()
                    except Exception:
                        inst_price = None
                    break
            # If no such instrument, fallback to the first candidate's price
            if inst_price is None and candidate_instruments:
                try:
                    inst_price = candidate_instruments[0].get_price()
                except Exception:
                    inst_price = None
            if inst_price is None or inst_price <= 0:
                inst_price = 100.0  # fallback

            cf_amt = sum(cf_by_mat[mat])

            if len(prev_maturities) == 0:
                # Zero-coupon: P = CF / (1+r)^T => r = (CF/P)^(1/T) - 1
                if cf_amt <= 0 or inst_price <= 0:
                    raise ValueError(f"Invalid cash flow or price at maturity {mat}")
                zero_rate = (cf_amt / inst_price) ** (1 / mat) - 1
                df = inst_price / cf_amt
            else:
                pv_prev = 0.0
                for i, prev_mat in enumerate(prev_maturities):
                    prev_cf_amt = sum(cf_by_mat[prev_mat])
                    pv_prev += prev_cf_amt * prev_dfs[i]
                if cf_amt == 0:
                    raise ValueError(f"Zero cash flow at maturity {mat}")
                df = (inst_price - pv_prev) / cf_amt
                if df <= 0:
                    zero_rate = 0.0
                else:
                    zero_rate = df ** (-1 / mat) - 1

            # Clamp zero_rate to a reasonable range (e.g., -0.99 to 1)
            zero_rate = min(max(zero_rate, -0.99), 1.0)

            self.maturities.append(mat)
            self.zero_rates.append(zero_rate)
            self.discount_factors.append(df)
            prev_maturities.append(mat)
            prev_dfs.append(df)

        self._create_interpolator()

    def _create_interpolator(self):
        if len(self.maturities) > 1:
            self.interpolator = interpolate.interp1d(
                self.maturities,
                self.zero_rates,
                kind=self.interp_method,
                bounds_error=False,
                fill_value="extrapolate"
            )
        else:
            self.interpolator = lambda x: self.zero_rates[0]

    def get_zero_rate(self, maturity):
        if self.interpolator is None:
            raise ValueError("Yield curve not bootstrapped yet. Call bootstrap() first.")
        if isinstance(maturity, (int, float)):
            if maturity < min(self.maturities):
                return self.zero_rates[0]
            elif maturity > max(self.maturities):
                return self.zero_rates[-1]
            else:
                return float(self.interpolator(maturity))
        else:
            return np.array([self.get_zero_rate(m) for m in maturity])

    def get_discount_factor(self, maturity):
        zero_rate = self.get_zero_rate(maturity)
        return 1 / ((1 + zero_rate) ** maturity)

    def get_forward_rate(self, start_maturity, end_maturity):
        if start_maturity >= end_maturity:
            raise ValueError("Start maturity must be less than end maturity")
        df_start = self.get_discount_factor(start_maturity)
        df_end = self.get_discount_factor(end_maturity)
        forward_period = end_maturity - start_maturity
        forward_rate = (df_start / df_end) ** (1 / forward_period) - 1
        return forward_rate

    def plot(self, title="Yield Curve"):
        if len(self.maturities) == 0:
            raise ValueError("Yield curve not bootstrapped yet. Call bootstrap() first.")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.maturities, self.zero_rates, color='blue', label='Actual Points')
        x_smooth = np.linspace(min(self.maturities), max(self.maturities), 100)
        y_smooth = self.get_zero_rate(x_smooth)
        ax.plot(x_smooth, y_smooth, linestyle='-', color='red', label='Interpolated Curve')
        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Zero Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        return fig

def date_to_year_fraction(date, base_date=None):
    if base_date is None:
        base_date = datetime.now()
    days_diff = (date - base_date).days
    year_fraction = days_diff / 365.0
    return year_fraction

def year_fraction_to_date(year_fraction, base_date=None):
    if base_date is None:
        base_date = datetime.now()
    days = int(year_fraction * 365)
    result_date = base_date + timedelta(days=days)
    return result_date

def extract_yield_curve_data(yield_curve, maturity_range=(0.1, 30), num_points=100):
    maturities = np.linspace(maturity_range[0], maturity_range[1], num_points)
    zero_rates = yield_curve.get_zero_rate(maturities)
    return maturities, zero_rates