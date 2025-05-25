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
        self.interp_method = 'linear' # Default interpolation method
        self.interpolator = None
        # Optional: Store initial rates if needed for fallback
        self.initial_rates_for_empty_curve = None


    def set_constituent_portfolio(self, portfolio):
        self.constituents = portfolio
        # Optional: If simulator passes initial rates for fallback
        if hasattr(portfolio, 'initial_rates_for_fallback'):
             self.initial_rates_for_empty_curve = portfolio.initial_rates_for_fallback

    def bootstrap(self):
        """
        Robust bootstrapping for a portfolio of zero-coupon and coupon bonds.
        Uses actual instrument prices.
        """
        if self.constituents is None:
            raise ValueError("No constituent instruments set. Use set_constituent_portfolio() first.")

        all_instruments = []
        if hasattr(self.constituents, 'bills') and self.constituents.bills:
            all_instruments.extend(self.constituents.bills)
        if hasattr(self.constituents, 'bonds') and self.constituents.bonds:
            all_instruments.extend(self.constituents.bonds)
        if hasattr(self.constituents, 'inflation_linked_bonds') and self.constituents.inflation_linked_bonds:
            all_instruments.extend(self.constituents.inflation_linked_bonds)

        if not all_instruments:
            # Fallback if no instruments are present
            if self.initial_rates_for_empty_curve:
                self.maturities = sorted(self.initial_rates_for_empty_curve.keys())
                self.zero_rates = [self.initial_rates_for_empty_curve[m] for m in self.maturities]
                self.discount_factors = [1 / ((1 + r)**m) for m, r in zip(self.maturities, self.zero_rates)]
            else: # Default fallback
                self.maturities = [0.1, 30]
                self.zero_rates = [0.01, 0.01] # Dummy flat 1%
                self.discount_factors = [1 / ((1 + 0.01)**0.1), 1 / ((1 + 0.01)**30)]
            self._create_interpolator()
            # Consider logging a warning here if using Streamlit:
            # import streamlit as st
            # st.warning("No instruments found for bootstrapping. Using fallback curve.")
            return

        # Sort instruments by maturity is crucial
        all_instruments.sort(key=lambda inst: inst.get_maturity())

        self.maturities = []
        self.zero_rates = []
        self.discount_factors = []

        temp_df_interpolator = None

        for inst in all_instruments:
            inst_maturity = inst.get_maturity()
            
            # Ensure instrument has price, often set after YTM
            if not hasattr(inst, 'price') or inst.price is None:
                # Attempt to calculate price if YTM is available (standard for non-bootstrapping context)
                # For bootstrapping, price should ideally be a market observation or pre-set
                if hasattr(inst, 'ytm') and inst.ytm is not None:
                    if not inst.cash_flows: inst.set_cash_flows() # Ensure cashflows are set
                    inst.price = inst.get_price() # Calculate price from YTM
                else:
                    raise ValueError(f"Instrument maturing at {inst_maturity} has no price and no YTM to derive it.")

            inst_price = inst.get_price() # This now uses the pre-set price from the simulator
            inst_cash_flows = inst.get_cash_flows() # Expected to be sorted by time

            # Price sanity check (can be adjusted)
            if not (0 < inst_price < 2000): # Increased upper bound for ILBs etc.
                raise ValueError(f"Invalid instrument price {inst_price:.2f} for instrument maturing at {inst_maturity:.2f}")

            pv_known_cash_flows = 0.0
            if temp_df_interpolator: # If we have prior DFs to use
                for t, cf_amount in inst_cash_flows:
                    if abs(t - inst_maturity) > 1e-8 and t < inst_maturity: # Coupon before final maturity
                        df_t = temp_df_interpolator(t)
                        if np.isnan(df_t): # Check if interpolator returned NaN (e.g. out of bounds with no extrapolation)
                             raise ValueError(f"Failed to get discount factor for intermediate cash flow at t={t} for instrument maturing at {inst_maturity}")
                        pv_known_cash_flows += cf_amount * df_t
            
            final_cash_flow_at_maturity = 0.0
            for t, cf_amount in inst_cash_flows:
                if abs(t - inst_maturity) < 1e-8: # Cash flow(s) exactly at maturity
                    final_cash_flow_at_maturity += cf_amount
            
            if final_cash_flow_at_maturity <= 1e-6: # Avoid division by zero or tiny numbers
                # This typically means an instrument definition issue (e.g. zero face value bond maturing)
                # Consider logging a warning with instrument details
                # import streamlit as st
                # st.warning(f"Instrument maturing at {inst_maturity:.2f} has zero or negligible final cash flow. Skipping.")
                continue # Skip this instrument

            pv_remaining = inst_price - pv_known_cash_flows

            # df can be > 1 for negative rates, but should not be <= 0
            if pv_remaining <= 1e-6 and final_cash_flow_at_maturity > 1e-6 : # Price implies near zero or negative value for final CF
                 # This situation means price is less than or equal to PV of known coupons.
                 # Leads to very high or undefined zero rates. Clamp DF to a small positive.
                 df = 0.01 # Arbitrary small positive DF, implies very high rate.
            elif final_cash_flow_at_maturity <= 1e-6 : # Should have been caught above
                 df = 0.01 # Should not happen if previous check is there
            else:
                 df = pv_remaining / final_cash_flow_at_maturity


            if df <= 1e-6: # If DF is zero or negative, also clamp (implies extremely high or infinite rate)
                df = 0.01 # Clamp to a small positive value

            # Calculate zero rate, ensure inst_maturity is not zero
            if abs(inst_maturity) < 1e-8:
                raise ValueError("Instrument maturity cannot be zero.")
            
            try:
                zero_rate = df ** (-1 / inst_maturity) - 1
            except (ValueError, OverflowError): # e.g. df is negative and inst_maturity is fractional
                 # This can happen if df was clamped to a small value due to pv_remaining issues
                 zero_rate = 2.0 # Clamp to a high rate like 200%
                 df = 1 / ((1 + zero_rate) ** inst_maturity)


            # Sanity check and clamp zero rates (e.g., -5% to 200%)
            # The original 0.1 was not a clamp boundary, so this is more for general stability
            sane_min_rate = -0.05 
            sane_max_rate = 2.00 # 200%
            if not (sane_min_rate <= zero_rate <= sane_max_rate):
                # import streamlit as st
                # st.warning(f"Zero rate {zero_rate:.4f} at mat {inst_maturity:.2f} (DF={df:.4f}, PVrem={pv_remaining:.2f}, FCF={final_cash_flow_at_maturity:.2f}, Prc={inst_price:.2f}) clamped.")
                zero_rate = min(max(zero_rate, sane_min_rate), sane_max_rate)
                df = 1 / ((1 + zero_rate) ** inst_maturity) # Recalculate DF from clamped rate


            # Avoid adding duplicate maturity points if instruments have exact same maturity
            # (though sorted list of unique instruments should mostly prevent this)
            if not self.maturities or abs(inst_maturity - self.maturities[-1]) > 1e-8:
                self.maturities.append(inst_maturity)
                self.zero_rates.append(zero_rate)
                self.discount_factors.append(df)
            elif abs(inst_maturity - self.maturities[-1]) < 1e-8 : # Same maturity as last point
                # Optionally, average or decide how to handle. For now, overwrite with new one if different.
                # Or log a warning: instruments with identical maturities might need averaging or a choice.
                # import streamlit as st
                # st.warning(f"Multiple instruments at maturity {inst_maturity:.2f}. Using last one.")
                self.zero_rates[-1] = zero_rate
                self.discount_factors[-1] = df


            # Update temporary interpolator for discount factors for subsequent instruments' coupons
            if len(self.maturities) == 1:
                # If only one point, extrapolate DFs using its zero rate
                current_zr = self.zero_rates[0]
                if abs(current_zr - (-1.0)) < 1e-6 : current_zr = -0.9999 # Avoid (1-1)=0
                temp_df_interpolator = lambda t_val: 1 / ((1 + current_zr) ** t_val)
            elif len(self.maturities) > 1:
                # Sort points for interpolator (should be sorted by insertion order, but defensive)
                # No, self.maturities are appended in order so they are already sorted.
                
                # Ensure unique maturities for interp1d
                unique_mats, unique_indices = np.unique(np.array(self.maturities), return_index=True)
                unique_dfs_for_interp = np.array(self.discount_factors)[unique_indices]

                if len(unique_mats) > 1:
                    # Boundary zero rates for extrapolation
                    first_zr_idx = np.argmin(unique_mats)
                    last_zr_idx = np.argmax(unique_mats)
                    
                    # Get the zero rate corresponding to the unique DFs for extrapolation
                    # This requires finding the original index of unique_dfs_for_interp in self.zero_rates via unique_mats
                    map_mat_to_zr = {m: zr for m, zr in zip(self.maturities, self.zero_rates)}
                    first_zr = map_mat_to_zr[unique_mats[first_zr_idx]]
                    last_zr = map_mat_to_zr[unique_mats[last_zr_idx]]
                    if abs(first_zr - (-1.0)) < 1e-6 : first_zr = -0.9999
                    if abs(last_zr - (-1.0)) < 1e-6 : last_zr = -0.9999

                    min_interp_mat = unique_mats[first_zr_idx]
                    max_interp_mat = unique_mats[last_zr_idx]

                    internal_coupon_df_interpolator = interpolate.interp1d(
                        unique_mats,
                        unique_dfs_for_interp,
                        kind='linear', # Linear interpolation for DFs between known points
                        bounds_error=False, # Allow querying outside, will use fill_value logic below
                        fill_value=np.nan # Placeholder, will be handled by wrapper
                    )
                    
                    def df_extrapolator_wrapper(t_val):
                        if t_val < min_interp_mat:
                            return 1 / ((1 + first_zr) ** t_val)
                        elif t_val > max_interp_mat:
                            return 1 / ((1 + last_zr) ** t_val)
                        else:
                            interpolated_val = internal_coupon_df_interpolator(t_val)
                            if np.isnan(interpolated_val) : # Should be covered by outer conditions but defensive
                                if t_val <= unique_mats[0] : return 1 / ((1 + first_zr) ** t_val)
                                else : return 1 / ((1 + last_zr) ** t_val)
                            return float(interpolated_val)
                    
                    temp_df_interpolator = df_extrapolator_wrapper
                
                elif len(unique_mats) == 1: # Should be caught by len(self.maturities)==1, but defensive
                    current_zr = self.zero_rates[0] # or map_mat_to_zr[unique_mats[0]]
                    if abs(current_zr - (-1.0)) < 1e-6 : current_zr = -0.9999
                    temp_df_interpolator = lambda t_val: 1 / ((1 + current_zr) ** t_val)


        if not self.maturities: # If loop finished but no points were added (e.g. all instruments skipped)
            # Fallback if no instruments resulted in valid curve points
            # import streamlit as st
            # st.error("Bootstrapping failed to produce any yield curve points after processing instruments. Using fallback.")
            if self.initial_rates_for_empty_curve:
                self.maturities = sorted(self.initial_rates_for_empty_curve.keys())
                self.zero_rates = [self.initial_rates_for_empty_curve[m] for m in self.maturities]
            else:
                self.maturities = [0.1, 30]
                self.zero_rates = [0.01, 0.01]
            self.discount_factors = [1 / ((1 + r)**m) for m, r in zip(self.maturities, self.zero_rates)]

        self._create_interpolator()

    def _create_interpolator(self):
        if not self.maturities: # No points at all
            # import streamlit as st
            # st.warning("Yield curve has no bootstrapped points. Interpolator will return a default rate (e.g., 1%).")
            self.interpolator = lambda x: 0.01 # Default flat rate
            return

        # Ensure maturities are sorted and unique for the final interpolator
        sorted_indices = np.argsort(self.maturities)
        sorted_mats = np.array(self.maturities)[sorted_indices]
        sorted_rates = np.array(self.zero_rates)[sorted_indices]

        unique_final_mats, unique_final_indices = np.unique(sorted_mats, return_index=True)
        unique_final_rates = sorted_rates[unique_final_indices]

        if len(unique_final_mats) > 1:
            self.interpolator = interpolate.interp1d(
                unique_final_mats,
                unique_final_rates,
                kind=self.interp_method, # Default 'linear'
                bounds_error=False,
                fill_value="extrapolate" # Standard extrapolation for final curve
            )
        elif len(unique_final_mats) == 1: # Only one unique point
            self.interpolator = lambda x: unique_final_rates[0] # Flat curve
        else: # Should be caught by the initial `if not self.maturities:`
            self.interpolator = lambda x: 0.01


    def get_zero_rate(self, maturity):
        if self.interpolator is None:
            # This can happen if bootstrap fails and _create_interpolator doesn't set a fallback
            raise ValueError("Yield curve not bootstrapped or interpolator not created. Call bootstrap() first.")
        
        # Handle single maturity or array of maturities
        if isinstance(maturity, (int, float, np.float64)):
            # Extrapolation for get_zero_rate is handled by interp1d's fill_value="extrapolate"
            # or by flat rates if only one point.
            # No need for manual boundary checks like in original code if interpolator handles it.
            rate = float(self.interpolator(maturity))
            # Final safety clamp on returned rates if desired, though clamping in bootstrap is primary
            # return max(min(rate, 2.0), -0.05) # e.g. -5% to 200%
            return rate
        else: # Assuming maturity is an array-like object
            # Ensure it's a NumPy array for the interpolator if it expects that
            rates = self.interpolator(np.array(maturity))
            # return np.maximum(np.minimum(rates, 2.0), -0.05)
            return np.array(rates, dtype=float)


    def get_discount_factor(self, maturity):
        zero_rate = self.get_zero_rate(maturity)
        # Handle cases where zero_rate could be -1, making (1+zero_rate) zero.
        if isinstance(zero_rate, np.ndarray):
            base = 1 + zero_rate
            base[base < 1e-6] = 1e-6 # Avoid division by zero if rate is -100%
            return 1 / (base ** maturity)
        else:
            base = 1 + zero_rate
            if base < 1e-6: base = 1e-6
            return 1 / (base ** maturity)

    def get_forward_rate(self, start_maturity, end_maturity):
        if np.any(start_maturity >= end_maturity): # Handle array inputs
            raise ValueError("Start maturity must be less than end maturity")
        
        df_start = self.get_discount_factor(start_maturity)
        df_end = self.get_discount_factor(end_maturity)
        
        forward_period = end_maturity - start_maturity
        
        # Avoid division by zero or negative DFs if they weren't fully sanitized
        if np.any(df_end <= 1e-6):
            # Handle error or return a placeholder for problematic forward rates
            # For arrays, this needs careful element-wise handling or a general approach
            # For now, assume DFs are positive. If not, an error or NaN would propagate.
            # A simple scalar check for illustration:
            if not isinstance(df_end, np.ndarray) and df_end <= 1e-6:
                 raise ValueError("Cannot compute forward rate with non-positive end discount factor.")

        ratio = df_start / df_end
        # Ensure ratio is positive for the root calculation
        if isinstance(ratio, np.ndarray):
            ratio[ratio < 1e-9] = 1e-9 # Clamp ratio to avoid issues with root of negative number
            # If period is zero (should be caught by start_maturity >= end_maturity)
            # forward_period[forward_period < 1e-9] = 1e-9
        elif ratio < 1e-9:
            ratio = 1e-9
        
        forward_rate = ratio ** (1 / forward_period) - 1
        return forward_rate

    def plot(self, title="Yield Curve"):
        if len(self.maturities) == 0:
            # import streamlit as st
            # st.warning("Cannot plot: Yield curve not bootstrapped or has no points.")
            # Optionally, plot a dummy/fallback curve if one was created
            if self.interpolator is None: return plt.figure() # Return empty figure

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_xlabel('Maturity (years)')
            ax.set_ylabel('Zero Rate')
            ax.set_title(f"{title} (No bootstrapped points - Fallback)")
            ax.grid(True)
            # Try to plot using interpolator if it's a fallback
            try:
                x_smooth = np.linspace(0.1, 30, 100) # Default range for fallback
                y_smooth = self.get_zero_rate(x_smooth)
                ax.plot(x_smooth, y_smooth, linestyle='--', color='gray', label='Fallback Interpolated Curve')
                if np.all(y_smooth == y_smooth[0]): # If flat
                     ax.legend([f"Fallback Flat Rate: {y_smooth[0]:.2%}"])
                else:
                     ax.legend()

            except Exception: # If interpolator itself is problematic
                 pass # Just show empty grid
            return fig


        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use unique, sorted points from the interpolator's creation if possible, or from self.maturities
        plot_mats = []
        plot_rates = []
        if hasattr(self.interpolator, 'x') and hasattr(self.interpolator, 'y'): # For scipy.interp1d
            plot_mats = self.interpolator.x
            plot_rates = self.interpolator.y
        else: # Fallback to raw bootstrapped points if interpolator is a lambda or other
            if self.maturities and self.zero_rates:
                sorted_indices = np.argsort(self.maturities)
                plot_mats = np.array(self.maturities)[sorted_indices]
                plot_rates = np.array(self.zero_rates)[sorted_indices]

        if len(plot_mats) > 0 :
             ax.scatter(plot_mats, plot_rates, color='blue', label='Actual Bootstrapped Points', zorder=5)
        
        # Generate smooth curve from the interpolator
        if len(plot_mats) > 0:
            min_mat_plot = max(0.01, np.min(plot_mats)) # Start plot from near zero or min maturity
            max_mat_plot = np.max(plot_mats)
            if max_mat_plot <= min_mat_plot : max_mat_plot = min_mat_plot + 10 # Default range if only one point
            
            # Extend plot range slightly for extrapolation visibility if desired, e.g., max_mat_plot * 1.1
            # Or use fixed range like 0 to 30 years if standard
            x_smooth = np.linspace(min_mat_plot, max_mat_plot, 200)
            y_smooth = self.get_zero_rate(x_smooth)
            ax.plot(x_smooth, y_smooth, linestyle='-', color='red', label='Interpolated Curve')
        elif self.interpolator : # No points but interpolator exists (e.g. fallback)
            x_smooth = np.linspace(0.1, 30, 100)
            y_smooth = self.get_zero_rate(x_smooth)
            ax.plot(x_smooth, y_smooth, linestyle='--', color='gray', label='Fallback Interpolated Curve')


        ax.set_xlabel('Maturity (years)')
        ax.set_ylabel('Zero Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        # Optional: Add y-axis formatting for percentage
        # from matplotlib.ticker import FuncFormatter
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.2%}'.format(y)))
        return fig

# Helper functions (date_to_year_fraction, etc.) remain unchanged
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