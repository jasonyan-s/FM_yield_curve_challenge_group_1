import importlib
import curve_classes_and_functions as yCurve
importlib.reload(yCurve)
import instrument_classes as inst
importlib.reload(inst)
import pandas as pd
import math
import matplotlib.pyplot as plt

def create_portfolio():
    # create a portfolio of two bank_bills
    yc_portfolio = inst.Portfolio()


    bank_bills = [
        inst.Bank_bill(maturity=0.25, ytm=0.047),
        inst.Bank_bill(maturity=0.50, ytm=0.050),
        inst.Bank_bill(maturity=0.75, ytm=0.053),
        inst.Bank_bill(maturity=1.00, ytm=0.050),
    ]
    
    for bill in bank_bills:
        bill.set_cash_flows()
        yc_portfolio.add_bank_bill(bill)


    bonds = [
        inst.Bond(maturity=1, coupon=0.055, ytm=0.056, frequency=2),
        inst.Bond(maturity=2, coupon=0.057, ytm=0.058,frequency=2),
        inst.Bond(maturity=5, coupon=0.060, ytm=0.062, frequency=1),
        inst.Bond(maturity=10, coupon=0.065, ytm=0.067,frequency=2),
    ]

    for bond in bonds:
        bond.set_cash_flows()
        yc_portfolio.add_bond(bond)

    yc_portfolio.set_cash_flows()
    print(yc_portfolio.get_cash_flows())

    return yc_portfolio

# print(yc_portfolio.get_cash_flows())

# create a yield curve based on the bank bill portfolio

# def yield_curve_construction(yc_portfolio):
#     yc=yCurve.YieldCurve()
#     yc.set_constituent_portfolio(yc_portfolio)
#     yc.bootstrap()
#     print(yc.get_zero_curve())


def main(): 
    yc2=yCurve.YieldCurve()
    yc_portfolio = create_portfolio()
    yc2.set_constituent_portfolio(yc_portfolio)
    yc2.bootstrap()
    print(yc2.get_zero_curve())
    return yc2.create_graph()

if __name__ == "__main__": 
    main()
    

