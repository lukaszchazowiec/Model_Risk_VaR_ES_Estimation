import yfinance as yf

# Pobranie danych historycznych dla S&P 500 (^GSPC)
df = yf.Ticker("^GSPC").history(start="2004-01-01", end="2025-12-31")

# Wyciągnięcie samej kolumny z cenami zamknięcia (Close)
zamkniecia = df['Close']

# Wyświetlenie kilku pierwszych wierszy
print(zamkniecia.head())

print(zamkniecia.shape)
print(zamkniecia.info)
print(zamkniecia.describe())
print(zamkniecia.isna().sum())