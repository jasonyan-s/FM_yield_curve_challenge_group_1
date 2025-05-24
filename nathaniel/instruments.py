class BankBill:
    def __init__(self, maturity, price):
        self.maturity = maturity  # in years
        self.price = price

    def get_maturity(self):
        return self.maturity

    def get_price(self):
        return self.price

class Bond:
    def __init__(self, maturity, coupon_rate, price):
        self.maturity = maturity  # in years
        self.coupon_rate = coupon_rate
        self.price = price

    def get_maturity(self):
        return self.maturity

    def get_price(self):
        return self.price
