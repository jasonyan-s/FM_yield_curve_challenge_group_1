import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time

# Set random seed for reproducibility
np.random.seed(42)

# ---------------------- Financial Instruments ----------------------

@dataclass
class BankBill:
    """A short‑term debt instrument"""
    face_value: float = 1_000_000  # $1M face value
    maturity_days: int = 90        # 90‑day bank bill
    price: float | None = None
    yield_rate: float | None = None

    def __post_init__(self):
        # Initialise missing fields
        if self.price is None and self.yield_rate is None:
            # Default to 5 % yield
            self.yield_rate = 0.05
            self.price = self.calculate_price_from_yield(self.yield_rate)
        elif self.price is None:
            self.price = self.calculate_price_from_yield(self.yield_rate)
        elif self.yield_rate is None:
            self.yield_rate = self.calculate_yield_from_price(self.price)

    # ---------- Pricing helpers ----------
    def calculate_price_from_yield(self, yield_rate: float) -> float:
        """Simple discount‑rate pricing."""
        return self.face_value / (1 + yield_rate * self.maturity_days / 365)

    def calculate_yield_from_price(self, price: float) -> float:
        return (self.face_value / price - 1) * 365 / self.maturity_days

    # ---------- Update helpers ----------
    def update_price(self, new_price: float):
        self.price = new_price
        self.yield_rate = self.calculate_yield_from_price(new_price)

    def update_yield(self, new_yield: float):
        self.yield_rate = new_yield
        self.price = self.calculate_price_from_yield(new_yield)

    # ---------- Pretty print ----------
    def __str__(self):
        return (
            f"BankBill(maturity={self.maturity_days}d, "
            f"price=${self.price:,.2f}, yield={self.yield_rate*100:.2f}%)"
        )


@dataclass
class Bond:
    """A longer‑term debt instrument"""
    face_value: float = 1_000_000
    coupon_rate: float = 0.05          # Annual coupon
    maturity_years: float = 5          # Years to maturity
    frequency: int = 2                 # Coupon frequency (semi‑annual)
    price: float | None = None
    yield_to_maturity: float | None = None

    def __post_init__(self):
        if self.price is None and self.yield_to_maturity is None:
            # Assume par pricing initially
            self.yield_to_maturity = self.coupon_rate
            self.price = self.calculate_price_from_ytm(self.yield_to_maturity)
        elif self.price is None:
            self.price = self.calculate_price_from_ytm(self.yield_to_maturity)
        elif self.yield_to_maturity is None:
            self.yield_to_maturity = self.calculate_ytm_from_price(self.price)

    # ---------- Pricing helpers ----------
    def calculate_price_from_ytm(self, ytm: float) -> float:
        periods = int(self.maturity_years * self.frequency)
        c = self.face_value * self.coupon_rate / self.frequency
        r = ytm / self.frequency

        pv_coupons = c * (1 - (1 + r) ** -periods) / r
        pv_principal = self.face_value / (1 + r) ** periods
        return pv_coupons + pv_principal

    def calculate_ytm_from_price(self, price: float, tol: float = 1e-10) -> float:
        low, high = 1e-4, 1.0
        while high - low > tol:
            mid = (low + high) / 2
            if self.calculate_price_from_ytm(mid) > price:
                low = mid
            else:
                high = mid
        return (low + high) / 2

    # ---------- Update helpers ----------
    def update_price(self, new_price: float):
        self.price = new_price
        self.yield_to_maturity = self.calculate_ytm_from_price(new_price)

    def update_ytm(self, new_ytm: float):
        self.yield_to_maturity = new_ytm
        self.price = self.calculate_price_from_ytm(new_ytm)

    def __str__(self):
        return (
            f"Bond(maturity={self.maturity_years}y, coupon={self.coupon_rate*100:.2f}%, "
            f"price=${self.price:,.2f}, YTM={self.yield_to_maturity*100:.2f}%)"
        )


@dataclass
class ForwardRateAgreement:
    """Contract on a future bank‑bill rate"""
    underlying_bill: BankBill
    settlement_days: int = 180
    price: float | None = None
    forward_rate: float | None = None

    def __post_init__(self):
        if self.price is None and self.forward_rate is None:
            self.forward_rate = self.calculate_theoretical_forward_rate()
            self.price = self.calculate_price_from_forward_rate(self.forward_rate)
        elif self.price is None:
            self.price = self.calculate_price_from_forward_rate(self.forward_rate)
        elif self.forward_rate is None:
            self.forward_rate = self.calculate_forward_rate_from_price(self.price)

    # ---------- Theory helpers ----------
    def calculate_theoretical_forward_rate(self) -> float:
        # Basic no‑arbitrage approximation (add 50 bp spread)
        return self.underlying_bill.yield_rate + 0.005

    def calculate_price_from_forward_rate(self, forward_rate: float) -> float:
        future_price = self.underlying_bill.face_value / (
            1 + forward_rate * self.underlying_bill.maturity_days / 365
        )
        discount = 1 / (1 + self.underlying_bill.yield_rate * self.settlement_days / 365)
        return future_price * discount

    def calculate_forward_rate_from_price(self, price: float) -> float:
        discount = 1 / (1 + self.underlying_bill.yield_rate * self.settlement_days / 365)
        future_price = price / discount
        return (
            (self.underlying_bill.face_value / future_price - 1)
            * 365 / self.underlying_bill.maturity_days
        )

    def calculate_arbitrage_opportunity(self) -> Tuple[bool, float]:
        theo_rate = self.calculate_theoretical_forward_rate()
        theo_price = self.calculate_price_from_forward_rate(theo_rate)
        diff = self.price - theo_price
        return abs(diff) > 1_000, diff

    # ---------- Update helpers ----------
    def update_price(self, new_price: float):
        self.price = new_price
        self.forward_rate = self.calculate_forward_rate_from_price(new_price)

    def update_forward_rate(self, new_rate: float):
        self.forward_rate = new_rate
        self.price = self.calculate_price_from_forward_rate(new_rate)

    def __str__(self):
        return (
            f"FRA(settle={self.settlement_days}d, price=${self.price:,.2f}, "
            f"fwd_rate={self.forward_rate*100:.2f}%)"
        )


@dataclass
class BondForward:
    """Contract on a future bond price"""
    underlying_bond: Bond
    settlement_days: int = 180
    price: float | None = None
    forward_yield: float | None = None

    def __post_init__(self):
        if self.price is None and self.forward_yield is None:
            self.forward_yield = self.calculate_theoretical_forward_yield()
            self.price = self.calculate_price_from_forward_yield(self.forward_yield)
        elif self.price is None:
            self.price = self.calculate_price_from_forward_yield(self.forward_yield)
        elif self.forward_yield is None:
            self.forward_yield = self.calculate_forward_yield_from_price(self.price)

    # ---------- Theory helpers ----------
    def calculate_theoretical_forward_yield(self) -> float:
        return self.underlying_bond.yield_to_maturity + 0.002

    def _adjusted_maturity(self) -> float:
        return self.underlying_bond.maturity_years - self.settlement_days / 365

    def calculate_price_from_forward_yield(self, forward_yield: float) -> float:
        temp_bond = Bond(
            face_value=self.underlying_bond.face_value,
            coupon_rate=self.underlying_bond.coupon_rate,
            maturity_years=self._adjusted_maturity(),
            frequency=self.underlying_bond.frequency,
            yield_to_maturity=forward_yield,
        )
        discount = 1 / (
            1 + self.underlying_bond.yield_to_maturity * self.settlement_days / 365
        )
        return temp_bond.price * discount

    def calculate_forward_yield_from_price(self, price: float) -> float:
        discount = 1 / (
            1 + self.underlying_bond.yield_to_maturity * self.settlement_days / 365
        )
        future_price = price / discount
        temp_bond = Bond(
            face_value=self.underlying_bond.face_value,
            coupon_rate=self.underlying_bond.coupon_rate,
            maturity_years=self._adjusted_maturity(),
            frequency=self.underlying_bond.frequency,
            price=future_price,
        )
        return temp_bond.yield_to_maturity

    def calculate_arbitrage_opportunity(self) -> Tuple[bool, float]:
        theo_yield = self.calculate_theoretical_forward_yield()
        theo_price = self.calculate_price_from_forward_yield(theo_yield)
        diff = self.price - theo_price
        return abs(diff) > 2_000, diff

    # ---------- Update helpers ----------
    def update_price(self, new_price: float):
        self.price = new_price
        self.forward_yield = self.calculate_forward_yield_from_price(new_price)

    def update_forward_yield(self, new_yield: float):
        self.forward_yield = new_yield
        self.price = self.calculate_price_from_forward_yield(new_yield)

    def __str__(self):
        return (
            f"BondFwd(settle={self.settlement_days}d, price=${self.price:,.2f}, "
            f"fwd_yield={self.forward_yield*100:.2f}%)"
        )


class YieldCurve:
    """Simple piece‑wise linear yield curve"""

    def __init__(self, bank_bills: List[BankBill], bonds: List[Bond]):
        self.bank_bills = sorted(bank_bills, key=lambda b: b.maturity_days)
        self.bonds = sorted(bonds, key=lambda b: b.maturity_years)
        self.maturities: List[float] = []
        self.yields: List[float] = []
        self.update_curve()

    def update_curve(self):
        self.maturities.clear()
        self.yields.clear()
        for bill in self.bank_bills:
            self.maturities.append(bill.maturity_days / 365)
            self.yields.append(bill.yield_rate)
        for bond in self.bonds:
            self.maturities.append(bond.maturity_years)
            self.yields.append(bond.yield_to_maturity)

    def get_interpolated_rate(self, maturity_years: float) -> float:
        if maturity_years <= self.maturities[0]:
            return self.yields[0]
        if maturity_years >= self.maturities[-1]:
            return self.yields[-1]
        for i in range(len(self.maturities) - 1):
            if self.maturities[i] <= maturity_years <= self.maturities[i + 1]:
                w = (
                    (maturity_years - self.maturities[i])
                    / (self.maturities[i + 1] - self.maturities[i])
                )
                return self.yields[i] + w * (self.yields[i + 1] - self.yields[i])

    def plot(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.maturities, [y * 100 for y in self.yields], "o-", lw=2)
        ax.set_xlabel("Maturity (years)")
        ax.set_ylabel("Yield (%)")
        ax.set_title("Yield Curve")
        ax.grid(True)
        return fig


class MarketSimulation:
    """Container holding all instruments and market logic"""

    def __init__(self):
        self.bank_bills = [
            BankBill(maturity_days=30, yield_rate=0.045),
            BankBill(maturity_days=60, yield_rate=0.047),
            BankBill(maturity_days=90, yield_rate=0.050),
            BankBill(maturity_days=180, yield_rate=0.053),
        ]
        self.bonds = [
            Bond(maturity_years=1, coupon_rate=0.055, yield_to_maturity=0.056),
            Bond(maturity_years=2, coupon_rate=0.057, yield_to_maturity=0.058),
            Bond(maturity_years=5, coupon_rate=0.060, yield_to_maturity=0.062),
            Bond(maturity_years=10, coupon_rate=0.065, yield_to_maturity=0.067),
        ]
        self.yield_curve = YieldCurve(self.bank_bills, self.bonds)
        self.fras = [
            ForwardRateAgreement(self.bank_bills[2], settlement_days=90),
            ForwardRateAgreement(self.bank_bills[2], settlement_days=180),
            ForwardRateAgreement(self.bank_bills[3], settlement_days=90),
        ]
        self.bond_forwards = [
            BondForward(self.bonds[0], settlement_days=90),
            BondForward(self.bonds[1], settlement_days=180),
            BondForward(self.bonds[2], settlement_days=90),
        ]

    def update_market(self, volatility: float = 0.05):
        # Random walk for yields
        for bill in self.bank_bills:
            bill.update_yield(max(0.001, bill.yield_rate + np.random.normal(0, volatility) * 0.003))
        for bond in self.bonds:
            bond.update_ytm(max(0.001, bond.yield_to_maturity + np.random.normal(0, volatility) * 0.004))
        self.yield_curve.update_curve()
        for fra in self.fras:
            deviation = np.random.normal(0, volatility) * 0.006
            fra.update_forward_rate(max(0.001, fra.calculate_theoretical_forward_rate() + deviation))
        for bf in self.bond_forwards:
            deviation = np.random.normal(0, volatility) * 0.007
            bf.update_forward_yield(max(0.001, bf.calculate_theoretical_forward_yield() + deviation))

    def get_arbitrage_opportunities(self) -> Dict[str, List[Dict]]:
        opps = {"fra": [], "bond_forward": []}
        for i, fra in enumerate(self.fras):
            has, diff = fra.calculate_arbitrage_opportunity()
            if has:
                opps["fra"].append(
                    {
                        "instrument": f"FRA {i+1}",
                        "description": f"Settlement {fra.settlement_days}d / Bill {fra.underlying_bill.maturity_days}d",
                        "market_price": fra.price,
                        "theoretical_price": fra.calculate_price_from_forward_rate(
                            fra.calculate_theoretical_forward_rate()
                        ),
                        "difference": diff,
                        "action": "Buy" if diff < 0 else "Sell",
                    }
                )
        for i, bf in enumerate(self.bond_forwards):
            has, diff = bf.calculate_arbitrage_opportunity()
            if has:
                opps["bond_forward"].append(
                    {
                        "instrument": f"Bond Forward {i+1}",
                        "description": f"Settlement {bf.settlement_days}d / Bond {bf.underlying_bond.maturity_years}y",
                        "market_price": bf.price,
                        "theoretical_price": bf.calculate_price_from_forward_yield(
                            bf.calculate_theoretical_forward_yield()
                        ),
                        "difference": diff,
                        "action": "Buy" if diff < 0 else "Sell",
                    }
                )
        return opps


# ---------------------- Streamlit App ----------------------

def main():
    st.set_page_config(page_title="Financial Market Simulator", layout="wide")

    # ---------- Title & intro ----------
    st.title("Dynamic Financial Market Simulator")
    st.markdown(
        """
        This application simulates a small fixed‑income market comprising:
        * **Bank Bills** – short‑term discount securities.
        * **Bonds** – longer‑term coupon‑paying debt.
        * **Forward Rate Agreements (FRAs)** and **Bond Forwards** – derivative contracts.

        Prices are randomly shocked each update; potential arbitrage opportunities are highlighted.
        """
    )

    # ---------- Session state ----------
    if "market_sim" not in st.session_state:
        st.session_state.market_sim = MarketSimulation()
        st.session_state.volatility = 0.5
        st.session_state.update_count = 0
        st.session_state.start_time = dt.datetime.now()
        # History containers
        st.session_state.price_history = {
            "bank_bills": {i: [] for i in range(len(st.session_state.market_sim.bank_bills))},
            "bonds": {i: [] for i in range(len(st.session_state.market_sim.bonds))},
            "fras": {i: [] for i in range(len(st.session_state.market_sim.fras))},
            "bond_forwards": {i: [] for i in range(len(st.session_state.market_sim.bond_forwards))},
        }
        st.session_state.yield_history: List[Tuple[List[float], List[float]]] = []
        st.session_state.timestamps: List[dt.datetime] = []
        st.session_state.previous_prices = {
            "bank_bills": [b.price for b in st.session_state.market_sim.bank_bills],
            "bonds": [b.price for b in st.session_state.market_sim.bonds],
            "fras": [f.price for f in st.session_state.market_sim.fras],
            "bond_forwards": [bf.price for bf in st.session_state.market_sim.bond_forwards],
        }

    # ---------- Layout ----------
    left, right = st.columns([1, 3])

    # ===== Left column – controls & summary =====
    with left:
        st.subheader("Market Controls")
        st.session_state.volatility = st.slider(
            "Market Volatility", 0.1, 1.0, st.session_state.volatility, 0.1
        )
        col_u, col_r = st.columns(2)
        with col_u:
            if st.button("Update Market", use_container_width=True):
                _update_market()
        with col_r:
            if st.button("Reset Simulation", use_container_width=True):
                st.session_state.clear()
                st.experimental_rerun()

        auto = st.checkbox("Auto‑update Market")
        interval = st.slider("Update Interval (s)", 1, 10, 3, disabled=not auto)
        st.markdown(
            f"Market Updates: **{st.session_state.update_count}**  •  "
            f"Running: **{(dt.datetime.now()-st.session_state.start_time).seconds}s**"
        )

        # ---------- Arbitrage summary ----------
        opps = st.session_state.market_sim.get_arbitrage_opportunities()
        total_opp = len(opps["fra"]) + len(opps["bond_forward"])
        st.subheader("Market Summary")
        st.metric("Arbitrage Opportunities", total_opp)

    # ===== Right column – visuals =====
    with right:
        tab_curve, tab_history = st.tabs(["Yield Curve", "Price History"])
        with tab_curve:
            st.pyplot(st.session_state.market_sim.yield_curve.plot())
            if len(st.session_state.yield_history) > 1:
                st.write("Yield Curve Evolution")
                fig, ax = plt.subplots(figsize=(10, 6))
                mats0, y0 = st.session_state.yield_history[0]
                mats1, y1 = st.session_state.yield_history[-1]
                ax.plot(mats0, [y*100 for y in y0], "o-", alpha=0.3)
                ax.plot(mats1, [y*100 for y in y1], "o-", lw=2)
                ax.set_xlabel("Maturity (years)")
                ax.set_ylabel("Yield (%)")
                ax.grid(True)
                st.pyplot(fig)
        with tab_history:
            if len(st.session_state.timestamps) < 2:
                st.info("Run a few updates to build history.")
            else:
                _plot_price_history()

    # ===== Market data tabs =====
    st.header("Live Market Data")
    t1, t2, t3, t4 = st.tabs(
        ["Bank Bills", "Bonds", "Forward Rate Agreements", "Bond Forwards"]
    )
    _render_bills(t1)
    _render_bonds(t2)
    _render_fras(t3)
    _render_bond_forwards(t4)

    # ===== Arbitrage dashboard =====
    st.header("Arbitrage Opportunities Dashboard")
    if total_opp == 0:
        st.info("No arbitrage opportunities at present.")
    else:
        if opps["fra"]:
            st.subheader("FRA Opportunities")
            st.table(pd.DataFrame(opps["fra"]))
        if opps["bond_forward"]:
            st.subheader("Bond Forward Opportunities")
            st.table(pd.DataFrame(opps["bond_forward"]))
        st.markdown(
            "**Strategy:**  Buy when market price < theoretical price;  "
            "Sell when market price > theoretical price."
        )

    # ===== Auto update loop =====
    if auto:
        time.sleep(interval)
        _update_market()
        st.experimental_rerun()


# ---------------------- Helper functions for Streamlit ----------------------

def _update_market():
    sim = st.session_state.market_sim
    # Preserve previous prices for change indicators
    st.session_state.previous_prices = {
        "bank_bills": [b.price for b in sim.bank_bills],
        "bonds": [b.price for b in sim.bonds],
        "fras": [f.price for f in sim.fras],
        "bond_forwards": [bf.price for bf in sim.bond_forwards],
    }
    sim.update_market(st.session_state.volatility)
    st.session_state.update_count += 1
    st.session_state.timestamps.append(dt.datetime.now())
    # History
    for i, b in enumerate(sim.bank_bills):
        st.session_state.price_history["bank_bills"][i].append(b.price)
    for i, b in enumerate(sim.bonds):
        st.session_state.price_history["bonds"][i].append(b.price)
    for i, f in enumerate(sim.fras):
        st.session_state.price_history["fras"][i].append(f.price)
    for i, bf in enumerate(sim.bond_forwards):
        st.session_state.price_history["bond_forwards"][i].append(bf.price)
    st.session_state.yield_history.append(
        (
            sim.yield_curve.maturities.copy(),
            sim.yield_curve.yields.copy(),
        )
    )


def _render_bills(container):
    with container:
        data = []
        for i, bill in enumerate(st.session_state.market_sim.bank_bills):
            prev = st.session_state.previous_prices["bank_bills"][i]
            change = bill.price - prev
            data.append(
                {
                    "Bill": f"{i+1}",
                    "Maturity (d)": bill.maturity_days,
                    "Price ($)": f"{bill.price:.2f}",
                    "Δ Price": f"{change:+.2f}",
                    "Yield (%)": f"{bill.yield_rate*100:.2f}",
                }
            )
        st.table(pd.DataFrame(data))


def _render_bonds(container):
    with container:
        data = []
        for i, bond in enumerate(st.session_state.market_sim.bonds):
            prev = st.session_state.previous_prices["bonds"][i]
            change = bond.price - prev
            data.append(
                {
                    "Bond": f"{i+1}",
                    "Maturity (y)": bond.maturity_years,
                    "Coupon (%)": f"{bond.coupon_rate*100:.2f}",
                    "Price ($)": f"{bond.price:.2f}",
                    "Δ Price": f"{change:+.2f}",
                    "YTM (%)": f"{bond.yield_to_maturity*100:.2f}",
                }
            )
        st.table(pd.DataFrame(data))


def _render_fras(container):
    with container:
        data = []
        for i, fra in enumerate(st.session_state.market_sim.fras):
            prev = st.session_state.previous_prices["fras"][i]
            change = fra.price - prev
            data.append(
                {
                    "FRA": f"{i+1}",
                    "Bill (d)": fra.underlying_bill.maturity_days,
                    "Settle (d)": fra.settlement_days,
                    "Price ($)": f"{fra.price:.2f}",
                    "Δ Price": f"{change:+.2f}",
                    "Fwd Rate (%)": f"{fra.forward_rate*100:.2f}",
                }
            )
        st.table(pd.DataFrame(data))


def _render_bond_forwards(container):
    with container:
        data = []
        for i, bf in enumerate(st.session_state.market_sim.bond_forwards):
            prev = st.session_state.previous_prices["bond_forwards"][i]
            change = bf.price - prev
            data.append(
                {
                    "BF": f"{i+1}",
                    "Bond (y)": bf.underlying_bond.maturity_years,
                    "Settle (d)": bf.settlement_days,
                    "Price ($)": f"{bf.price:.2f}",
                    "Δ Price": f"{change:+.2f}",
                    "Fwd Yld (%)": f"{bf.forward_yield*100:.2f}",
                }
            )
        st.table(pd.DataFrame(data))


def _plot_price_history():
    instruments = st.radio(
        "Select Instrument Type", ["Bank Bills", "Bonds", "FRAs", "Bond Forwards"], horizontal=True
    )
    mapping = {
        "Bank Bills": ("bank_bills", "Bank Bill Price History"),
        "Bonds": ("bonds", "Bond Price History"),
        "FRAs": ("fras", "FRA Price History"),
        "Bond Forwards": ("bond_forwards", "Bond Forward Price History"),
    }
    key, title = mapping[instruments]
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, series in st.session_state.price_history[key].items():
        if series:
            ax.plot(range(len(series)), series, "-o", label=f"{instruments[:-1]} {i+1}")
    ax.set_xlabel("Market Updates")
    ax.set_ylabel("Price ($)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
