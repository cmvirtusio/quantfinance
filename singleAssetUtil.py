import math as math
import numpy as np
from datetime import datetime
from datetime import timedelta
import pandas_datareader as pdr

def get_period_returns(dataframe):
    array = []
    prev_adj_close_price = float('nan')
    for close_price in dataframe['Adj Close'].iteritems():
        if not math.isnan(prev_adj_close_price):
            # array[close_price[0]] = (close_price[1] / prev_adj_close_price - 1)
            array.append((close_price[1] / prev_adj_close_price - 1))
        prev_adj_close_price = close_price[1]

    return array

def get_num_of_shares_per_unit(ticker_returns, ticker_price, period_length, daily_risk_per_asset, symbol):
    daily_risk = daily_risk_per_asset
    stdev = np.std(ticker_returns)
    s = pdr.data.get_data_yahoo(symbols=symbol, start=datetime.now().date() - timedelta(days=60),
                                end=datetime.now().date(), interval='d')
    atr = getATR(s, period_length)
    return math.floor(daily_risk / (3 * max(atr, stdev * ticker_price)))

def get_notional_value_per_unit(ticker_returns, ticker_price, period_length, daily_risk_per_asset, symbol):
    return get_num_of_shares_per_unit(ticker_returns, ticker_price, period_length, daily_risk_per_asset, symbol) * ticker_price

def getATR(symbolTickerPrices, periodLength):
    trueRangeSum = 0

    for x in range (1, periodLength):
        dailyValue = symbolTickerPrices.iloc[x]
        prevDailyValue = symbolTickerPrices.iloc[x-1]
        trueRange1 = abs(dailyValue['High'] - dailyValue['Low'])
        trueRange2 = abs(dailyValue['High'] - prevDailyValue['Close'])
        trueRange3 = abs(dailyValue['Low'] - prevDailyValue['Close'])

        trueRangeSum += max(trueRange1, trueRange2, trueRange3)

    return trueRangeSum/periodLength

def getRollingATR(dataframe, timeInterval, periodLength, rollingLength):
    # df = pdr.data.get_data_yahoo(symbols=symbolTicker, start=datetime.now().date() - timedelta(days=timeInterval),
    #                              end=datetime.now().date(), interval='d')

    retArray = []

    for x in range(0, rollingLength):
        retArray.append(getATR(dataframe[x:], periodLength))
    return retArray

def getVolatility(arrayOfPercentReturns, lengthPeriod):
    return np.std(arrayOfPercentReturns[:lengthPeriod])

def getRollingVolatility(lengthPeriod, rollPeriod, tickerSymbol):
    array = get_period_returns(pdr.data.get_data_yahoo(symbols=tickerSymbol, start=datetime.now().date() - timedelta(days=lengthPeriod),
                                     end=datetime.now().date(), interval='d'))
    retArray = []

    for x in range(0, rollPeriod):
        retArray.append(getVolatility(array[x:], lengthPeriod))
    return retArray