import yfinance as yf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime

# define instruments to download
# define instruments to download
companies_dict = {}
data_csv = pd.read_excel('holdings-daily-us-en-spy.xlsx').dropna()
companies_dict = data_csv["Ticker"].values.tolist()

data = []
for stock in companies_dict:
    try:
        info = yf.Ticker(stock).info
        all_shares = info.get('sharesOutstanding')
        prev_close = info.get('previousClose')
        market_cap = all_shares * prev_close
        yahoo_market_cap = info.get('marketCap')
        ebitda = info.get('ebitda')
        ebitda_ratio = market_cap / ebitda
        trail_pe_ratio = info.get('trailingPE')
        pb_ratio = info.get('priceToBook')
        debt_to_equity = info.get('debtToEquity')
        free_cash_flow = info.get('freeCashflow')
        peg_ratio = info.get('pegRatio')
    except:
        print('Error with stock: ', stock)
        continue

    
    print(stock, all_shares, prev_close, market_cap,yahoo_market_cap, ebitda, ebitda_ratio)
    data.append({
        'Company Name': list(companies_dict)[list(companies_dict).index(stock)],
        'Stock Ticker Symbol': stock,
        'Number of Outstanding Shares': all_shares,
        'Previous Close Price': prev_close,
        'Market Capitalization': market_cap,
        'Yahoo Market Capitalization': yahoo_market_cap,
        'EBITDA': ebitda,
        'EBITDA Ratio': ebitda_ratio,
        'TR PE Ratio': trail_pe_ratio,
        'PB Ratio': pb_ratio,
        'Debt to Equity': debt_to_equity,
        'Free Cash Flow': free_cash_flow,
        'PEG Ratio': peg_ratio
    })

df = pd.DataFrame(data)
df.to_csv('valuation_examples.csv')
#EBTIDA value is above 10, which is considered healthy and above average by analysts and investors.
#EBTIDA value is calculated by dividing the market capitalization by the EBITDA. The market capitalization is the total value of all of a company's shares of stock. The EBITDA is the earnings before interest, taxes, depreciation, and amortization.
#The EBTIDA metric is used to determine the value of a company. It is also used to compare a company's value to its competitors' values. 
# A company with a high EBITDA value is considered to be a good investment because it has a high value and is worth more than its competitors.
#A company with a low EBITDA value is considered to be a bad investment because it has a low value and is worth less than its competitors.
#A good EBITDA baseline for a company with a market capitalization of under $1 billion is a ratio of 10 or more.
#For a company with a market capitalization over $1 billion, a ratio of 8 is a good baseline.
#As a general guideline, an EV/EBITDA value below 10 is commonly interpreted as healthy and above average
#by analysts and investors. A value above 10 could indicate that a company is overvalued or that its assets
#are generating low returns. In some cases, a low EV/EBITDA ratio may be a sign that a company is undervalued
#or that its stock price is too low.
#A P/B ratio of 0.95, 1, or 1.1 means the underlying stock is trading at nearly book value. In other words, the P/B 
# ratio is more useful the greater the number differs from 1. To a value-seeking investor, a company that trades for a P/B ratio of 0.5 
# is attractive because it implies that the market value is one-half of the company's stated book value. 
#print(df.head())
#msft = yf.Ticker(companies_dict.get(companies_dict.keys().__iter__().for_next()))
#print(msft.info['sharesOutstanding'])
#print(msft.info['previousClose'])
#print(msft.info['sharesOutstanding'] * msft.info['previousClose'])
#print(msft.info['marketCap'])

# multiply the number of outstanding shares by the current market value of one share
#msft_shares_outstanding = msft.info['sharesOutstanding']
#msft_market_cap = msft.info['marketCap']
#msft_shares_outstanding * msft_market_cap

#ticker = list(companies_dict.values())
#start = datetime.datetime.now() - datetime.timedelta(days=5)
#end = datetime.datetime.now()

# Download the historical stock prices
#data = yf.download(ticker, start=start, end=end, interval='1h', auto_adjust=True)
#panel_data = data.dropna()