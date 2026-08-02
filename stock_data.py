import yfinance as yf

ticker = yf.Ticker("AAPL")
info = ticker.fast_info

print("Last price:", info.last_price)
print("Previous close:", info.previous_close)
