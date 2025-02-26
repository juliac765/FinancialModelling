import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import plot  # Import the plot function
import os
# Download stock data
tickers = ['AAPL', 'MC.PA', 'UNH']
print(os.getcwd())

#%%
data = yf.download(tickers, start='2024-01-01', end='2024-12-01', 
                   auto_adjust = False, multi_level_index=False)
#%%
#  Correctly extract Adjusted Close using MultiIndex column selection
adj_close = data.loc[:, ('Adj Close', slice(None))]

# Rename columns to remove MultiIndex for easier handling
adj_close.columns = adj_close.columns.droplevel(0)

#  Compute returns for each ticker
returns = adj_close.pct_change().dropna()

#  Plot Adjusted Close Prices for all tickers separately
adj_close.plot(title="Adjusted Close Prices", figsize=(10, 6))
plt.ylabel('Price')
plt.grid(True)
plt.show()

#%%
# Cumulative returns and metrics
cumulative_returns = (1 + returns).cumprod()
annualized_return = returns.mean() * 252
annualized_volatility = returns.std() * np.sqrt(252)

# Print simple metrics
print(f"Annualized Return:\n{annualized_return}\n")
print(f"Annualized Volatility:\n{annualized_volatility}\n")
print(cumulative_returns.columns)


# Sharpe ratio
risk_free_rate = 0.03 / 252
sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()

#%%
sharpe_ratio.plot(kind = 'bar', title = 'Shape Ratios', color = 'skyblue', figsize = (10,6))
plt.ylabel('Sharpe Ratio')
plt.grid(True)
plt.show()

#%%
returns_long = returns.melt(var_name = 'Stock', value_name = 'Daily Return') #reshapes the dataframe for use within Plotly
fig = px.histogram(returns_long, x = 'Daily Return', color = 'Stock',
                   facet_col = 'Stock', facet_col_wrap=3, nbins = 50,
                   color_discrete_sequence= px.colors.qualitative.Set2)
fig.update_layout(title_text = 'Daily Returns Distribution',
                  height = 600, width = 1000)
plot(fig)

#%%
weights = np.array([1/3,1/3,1/3])
portfolio_returns = returns.dot(weights)
cumulative_portfolio_returns = (1+portfolio_returns).cumprod()

#%%
plt.figure(figsize = (10,6))
plt.plot(cumulative_portfolio_returns, label = "Portfolio", color = 'purple')
plt.title("Cumulative Portfolio Returns")
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.legend()
plt.show()

#%%
cumulative_returns['Portfolio'] = cumulative_portfolio_returns

cumulative_returns = cumulative_returns.reset_index()

print(cumulative_returns.columns)

#%%
fig = px.line(cumulative_returns, x= 'Date', y=['AAPL', 'MC.PA', 'UNH', 'Portfolio'],
              title = 'Cumulative Returns: Portfolio Assets')
fig.update_layout(yaxis_title = 'Cumulative Return', xaxis_title = 'Date')
plot(fig)


