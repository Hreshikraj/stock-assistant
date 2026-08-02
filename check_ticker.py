import yfinance as yf

ticker = yf.Ticker("AAPL")
info = ticker.fast_info
print(info.last_price)