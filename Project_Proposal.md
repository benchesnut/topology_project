Time Signal Analysis of Stock Price Trends
===================



####Group members: Ben Chesnut, Raghav Kedia, Jack Gillette

Goals/Questions Asked:
-------------
Our project will focus on using techniques from topological data analysis to study features and trends in stock prices, and to potentially create a predictive model based on these trends to guide the purchase and selling of stocks. Specifically, we want to see if specific features are more relevant to certain trends and industries, and if similar feature patterns between past and current stock prices correlate with similar trends. We also want to investigate the optimal time scale for which these patterns and trends occur (minute scale, daily scale, weekly scale, etc…).

Mathematical Methods:
-------------
In order to study the change of price through time, we will use sliding window embedding to generate point clouds based on various features and technical indicators. We will then use Rips filtration on these point clouds and study the persistence of the resulting simplicial complexes. By varying the technical indicators used as well as the size and time step of the sliding windows, we will generate many different persistence measurements for each index. From these persistence measurements, we hope to gain meaningful insights about stock trends, such as which technical indicators are more relevant or indicative for certain trends and for certain stocks and industries, and to be able to use these insights to accurately predict increases and decreases in prices.

Datasets
-------------
Quantopian: https://www.quantopian.com/home
Quandl: https://blog.quandl.com/api-for-global-stock-data

We will focus primarily on US equities and US indices. For our datasets, we will be using an API from Quantopian. Quantopian is a “crowd-sourced” hedge fund, with a feature that allows users to to backtest trading strategies for free. Quantopian has minute bar equity data starting from January 2002. By having access to minute bar data, this will give us the flexibility to experiment with a variety of different time windows. Quantopian also gives us the ability to build trading strategies and test them, so we will be able to build some algorithms based on the insights we gain from our analysis. We would also like to perform analysis on International markets. We will use Quandl’s API to pull data on international equities. Quandl only gives us access to end-of-day prices, so this may limit our analysis in terms of time windows.

###Instruments we will focus on (subject to change):
####Equities:
Apple (AAPL)
Google (GOOG)
ExxonMobile (XOM)
Celgene Corp (CELG)
International Equities (to be determined)

####Indices:
S&P 500
DJIA


####Specific Datasets:
- All stocks in S&P 500 (daily and possibly minute) pulled from Quandl

####Features for each stock
- Raw stock prices/indexes
- Mean-centered stock prices
- Beta (relative measure of volatility)
- RSI
- 5-day moving average
- Average Directional Index (ADX)


####Different Methods:
- Weierstrauss Distance to compare different stocks (Bottleneck Stability), Mean-center stocks, compare by windows
- Sliding Window Delayed Reconstruction to find patterns in raw individual stock prices
