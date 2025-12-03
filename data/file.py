import yfinance as yf

# Download Apple stock data (AAPL)
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01")

# Show first 5 rows
print(data.head())

# Save to CSV
data.to_csv("data/raw/apple_stock.csv")
