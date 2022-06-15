# use golden cross as buy in signal
# Sell when short timeframe < max
# Buy back if short > max

# sell when down x% from max
# buying the dip: if coin drops 5-10% outside of historical average standard price movement, buy. Sell at 5% gain
import main
import ta
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from scipy import stats
from scipy.signal import savgol_filter
from datetime import datetime, timedelta
from time import time
import pandas as pd

api_key = "8BVVt4nlaB77zzuRtHLdVvQS7wemQQI1mIQDc9pENhIZCHZNYWVs3CwA9QTEyb5d"
api_secret = "Fuf2USPB2UY372WbULiuh265R3FIqBoL2IsFbUFX8wzvy2nG68m3znzPg6sCWj7k"
client = Client(api_key, api_secret)


def get_quote_coins(base):
    coins = []
    all_tickers = client.get_ticker()
    for ticker in all_tickers:
        if ticker['symbol'][-len(base):] == base and float(ticker['volume']) * float(ticker['weightedAvgPrice']) > 5000000:
            coins.append(ticker['symbol'][:-len(base)].upper())
    return coins


def buy(coin, price, wallet, fees):
    wallet[coin] = (wallet['USDT'] / price) * fees
    wallet['USDT'] = 0
    return wallet


def sell(coin, price, wallet, fees):
    if wallet[coin] > 0:
        wallet['USDT'] = wallet[coin] * price * fees
        wallet[coin] = 0
    return wallet


def main_sim(data, coin='BTC', fees=0.9985, wallet=None, output=False):
    if not wallet:
        wallet = {'USDT': 10000, coin: 0}
    for index, row in data.iterrows():
        if row['Pos Crossing']:
            wallet = buy(coin, row['Close'], wallet, fees)
        elif row['Neg Crossing']:
            wallet = sell(coin, row['Close'], wallet, fees)

    if wallet['USDT'] == 0:
        wallet = sell(coin, row['Close'], wallet, fees)
    return wallet


def n_days_ago(n):
    return (datetime.today() - timedelta(days=n)).strftime("%Y %m %d")


def init(coin='BTC', start_date=n_days_ago(3), end_date=n_days_ago(1), interval='5m'):

    data = main.get_hist_data(coin=coin, start_date=start_date, end_date=end_date, interval=interval)
    data['Close'] = ta.trend.sma_indicator(data['Close'], window=1)

    data['Price Movement'] = (data['Close'].diff() / data['Close'])*100
    data['SMA 7'] = ta.trend.sma_indicator(data['Close'], window=7)
    data['EMA 7'] = ta.trend.ema_indicator(data['Close'], window=7)
    data['SMA 25'] = ta.trend.sma_indicator(data['Close'], window=25)

    # data['sav gol'] = main.savitzky_golay(data['Close'], 51, 3)  # window size 51, polynomial order 3
    data['sav gol'] = savgol_filter(data['Close'], 25, 3)
    # data['SMA 25 tangent line'] =

    ''' 
    p7 = data['SMA 7'].shift(1)
    p25 = data['SMA 25'].shift(1)

    data['Pos Crossing'] = (data['SMA 7'] >= data['SMA 25']) & (p7 <= p25)
    data['Neg Crossing'] = (data['SMA 7'] <= data['SMA 25']) & (p7 >= p25)
    '''

    # buy when SMA 7 crosses through sav gol and slope of tangent of SMA 25 is positive
    sav_gol = data['sav gol'].shift(1)
    p7 = data['SMA 7'].shift(1)

    data['Pos Crossing'] = (data['sav gol'] >= data['SMA 7']) & (sav_gol <= p7)
    data['Neg Crossing'] = (data['sav gol'] <= data['SMA 7']) & (sav_gol >= p7)

    #
    '''
    slope, intercept, r_value, p_value, std_err = stats.linregress(data['open_time'], data['Close'])
    data['slope'] = slope
    data['intercept'] = intercept
    data['r_value'] = r_value
    data['p_value'] = p_value
    data['std_err'] = std_err

    print(24, data['Price Movement'].max(), data['Price Movement'].min(), data['Price Movement'].std())
    print(12, data['Price Movement'].tail(60*12).max(), data['Price Movement'].tail(60*12).min(), data['Price Movement'].tail(60*12).std())
    print(6, data['Price Movement'].tail(60*6).max(), data['Price Movement'].tail(60*6).min(), data['Price Movement'].tail(60*6).std())
    print(1, data['Price Movement'].tail(60).max(), data['Price Movement'].tail(60).min(), data['Price Movement'].tail(60).std())
    '''

    data.to_csv('data.csv', index=None, header=True)
    plt.plot(data['Close'], label='Close')
    # plt.plot(data['EMA 7'], label='EMA 7')
    plt.plot(data['SMA 7'], label='SMA 7')
    plt.plot(data['SMA 25'], label='SMA 25')
    plt.plot(data['sav gol'], label='Sav Gol')
    plt.legend()
    plt.savefig('data.pdf', format='pdf')
    return data


if __name__ == "__main__":
    start_sim_t = time()
    coins_ = get_quote_coins('USDT')
    print(coins_)
    data_ = {}
    for coin_ in coins_:
        data_[coin_] = main_sim(init(coin_, start_date=n_days_ago(30), end_date=n_days_ago(1), interval='5m'), coin=coin_)['USDT']
        print({coin_: data_[coin_]})
    print(data_)
    print('Runtime:', str(time()-start_sim_t))