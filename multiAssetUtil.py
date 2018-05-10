from datetime import datetime
from datetime import timedelta
import pandas_datareader as pdr
import pandas_datareader.oanda as oanda


def getAssetDataAndSanitize(tickers, quote_currency):
    # tickers = ['SPY', 'GLD', 'TLT', 'HVI.TO', 'CTC.TO', 'TD', 'DBA', 'XOM', 'BBD-B.TO', 'BBRY', 'GBP', 'EUR']
    # quote_currency = 'USD'

    ticker_data = []

    for x in range(0, len(tickers)):
        try:
            df = pdr.data.get_data_yahoo(symbols=tickers[x], start=datetime.now().date() - timedelta(days=120),
                                         end=datetime.now().date(), interval='d')

            ticker_data.append(df)
        except:
            df = oanda.get_oanda_currency_historical_rates(datetime.now().date() - timedelta(days=120),
                                                           datetime.now().date(),
                                                           quote_currency=quote_currency, base_currency=tickers[x])
            ticker_data.append(df)

    for x in range(0, len(ticker_data)):
        for y in range(0, len(ticker_data)):
            (ticker_data[x], ticker_data[y]) = ticker_data[x].align(ticker_data[y], join='inner', axis=0)

    return ticker_data
