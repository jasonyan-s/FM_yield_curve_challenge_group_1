import numpy as np
import math

def exp_interp(xs, ys, x):
    xs = np.array(xs)
    ys = np.array(ys)
    idx = np.searchsorted(xs, x) - 1
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    rate = (math.log(y1) - math.log(y0)) / (x1 - x0)
    return y0 * math.exp(rate * (x - x0))

class ZeroCurve:
    def __init__(self):
        self.maturities = []
        self.zero_rates = []
        self.AtMats = []
        self.discount_factors = []

    def add_zero_rate(self, maturity, zero_rate):
        self.maturities.append(maturity)
        self.zero_rates.append(zero_rate)
        self.AtMats.append(math.exp(zero_rate * maturity))
        self.discount_factors.append(1 / self.AtMats[-1])

    def add_discount_factor(self, maturity, df):
        self.maturities.append(maturity)
        self.discount_factors.append(df)
        self.AtMats.append(1 / df)
        self.zero_rates.append(math.log(1 / df) / maturity)

    def get_AtMat(self, m):
        if m in self.maturities:
            return self.AtMats[self.maturities.index(m)]
        return exp_interp(self.maturities, self.AtMats, m)

    def get_discount_factor(self, m):
        if m in self.maturities:
            return self.discount_factors[self.maturities.index(m)]
        return exp_interp(self.maturities, self.discount_factors, m)

    def get_zero_rate(self, m):
        if m in self.maturities:
            return self.zero_rates[self.maturities.index(m)]
        return math.log(self.get_AtMat(m)) / m

    def get_zero_curve(self):
        return self.maturities, self.discount_factors

    def npv(self, cash_flows):
        return sum(
            cash_flows.get_cash_flow(m) * self.get_discount_factor(m)
            for m in cash_flows.get_maturities()
        )

class YieldCurve(ZeroCurve):
    def __init__(self):
        super().__init__()
        self.portfolio = None

    def set_constituent_portfolio(self, portfolio):
        self.portfolio = portfolio

    def bootstrap(self):
        self.add_zero_rate(0, 0)
        for bb in self.portfolio.get_bank_bills():
            self.add_discount_factor(
                bb.get_maturity(),
                bb.get_price() / bb.get_face_value()
            )
        for b in self.portfolio.get_bonds():
            pv = 0
            dates, amts = b.get_maturities(), b.get_amounts()
            for i in range(1, len(amts) - 1):
                pv += amts[i] * self.get_discount_factor(dates[i])
            self.add_discount_factor(
                b.get_maturity(),
                (b.get_price() - pv) / amts[-1]
            )