import yfinance as yf
import ollama

# Step 1: get real data
ticker = yf.Ticker("AAPL")
info = ticker.fast_info
last_price = info.last_price
previous_close = info.previous_close

# Step 2: build a prompt that includes that real data
prompt = f"""
AAPL's last traded price is {last_price:.2f}.
Its previous close was {previous_close:.2f}.

Based on this data, briefly explain whether the stock is up or down today, and by how much (in dollars and percent).
"""

# Step 3: send it to the LLM
response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": prompt}]
)

print(response["message"]["content"])