import yfinance as yf
t = yf.Ticker("AAPL")
cashflow = t.cashflow
print(cashflow.loc["Free Cash Flow"])
