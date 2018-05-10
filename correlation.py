import math
import pandas as pd


def get_correlation_of_change_of_adj_close(array_1, array_2):
    array_diff_1 = {}
    array_diff_2 = {}

    prev_adj_close_price = float('nan')
    for close_price in array_1:
        if not math.isnan(prev_adj_close_price):
            array_diff_1[close_price[0]] = close_price[1] - prev_adj_close_price
        prev_adj_close_price = close_price[1]

    prev_adj_close_price = float('nan')
    for close_price in array_2:
        if not math.isnan(prev_adj_close_price):
            array_diff_2[close_price[0]] = close_price[1] - prev_adj_close_price
        prev_adj_close_price = close_price[1]

    a1 = pd.Series(array_diff_1)
    a2 = pd.Series(array_diff_2)

    return a1.corr(a2)


def get_period_returns(dataframe):
    array = []
    prev_adj_close_price = float('nan')
    for close_price in dataframe['Adj Close'].iteritems():
        if not math.isnan(prev_adj_close_price):
            # array[close_price[0]] = (close_price[1] / prev_adj_close_price - 1)
            array.append((close_price[1] / prev_adj_close_price - 1))
        prev_adj_close_price = close_price[1]

    return array


def get_rolling_correlation(array_1, array_2, correlation_data_length, roll_period):
    # roll_period is the period over which the correlation is done over
    # correlation_data_length is the number of correlations to return e.g. 30 is for the last 30 days
    cor_array = []
    combined_df = array_1.join(array_2, lsuffix='_caller', rsuffix='_other').dropna()
    length = len(combined_df)
    for i in range(0, roll_period):
        index = i + (length - roll_period)
        cor_array.append(get_correlation_of_change_of_adj_close(
            combined_df.iloc[(index - correlation_data_length):index]['Adj Close_caller'].iteritems(),
            combined_df.iloc[(index - correlation_data_length):index]['Adj Close_other'].iteritems()))

    cor_array.reverse()
    return cor_array
