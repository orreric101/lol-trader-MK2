import pandas as pd
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
import asyncio
import datetime as dt
import matplotlib.pyplot as plt
from time import time, sleep
import requests
import numpy as np
from math import factorial


api_key = "5zt3nLqIbfc4b2g0iT1aiI0rxWrD16KNNETF3Fc3xyVDORP0s4J44bcObkNQHgKN"
api_secret = "7G6wg0NhGtrSIGtX8FXEAlgia3I6pFVPj6sa52nvUdgaV6tCyuHJ3OZkZBb1N08R"
client = Client(api_key, api_secret, testnet=True)

TIMEFRAME = 15 # Long I keep the data


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def get_hist_data(coin='BTC', interval='1m', start_date='1 Jan,2021', end_date='31 Dec, 2021', path=''):
    # print('Getting '+coin+' Data')
    t1 = time()
    symbol = coin + 'USDT'
    klines = client.get_historical_klines(symbol, interval, start_date, end_date)
    data = pd.DataFrame(klines)

    # create columns name
    data.columns = ['open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'qav', 'num_trades',
                    'taker_base_vol', 'taker_quote_vol', 'ignore']

    # change the timestamp
    data.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in data.close_time]
    # path = 'coins database 2021/'+coin+'.csv'
    # data.to_csv(path, index=None, header=True)

    # convert data to float and plot
    '''
    df = data.astype(float)
    df["close"].plot(title=symbol, legend='close')
    plt.show()

    print('Done ', coin)
    print('Runtime:', time() - t1)
    '''
    return data


def get_curr_data():
    print(client.get_klines(symbol='BTCUSDT', interval='1m', limit=1))


def pre_get_hist_data():
    x = 0
    for coin in coins:
        get_hist_data(coin=coin, start_date='Jan 1, 2021', end_date='Dec 31, 2021', interval='1m')
        x += 1
        print('Coins:', x)


def analyze_price_movement(data):
    first_price, last_price = float(data[0]['p']), float(data[-1]['p'])
    change = ((last_price / first_price)-1)*100
    print(f'{change:.20f}', '%')


def trim_data(data):
    # print(data[0]['E'], (time() - 5)*1000)
    if data[0]['E'] < ((time() - TIMEFRAME)*1000):
        del data[0]
        trim_data(data)
    return data


async def open_socket(symbols):
    bm_client = await AsyncClient.create()
    bm = BinanceSocketManager(bm_client)
    # socket = bm.trade_socket(symbol)
    socket = bm.multiplex_socket(symbols)
    # socket = bm.miniticker_socket()

    data = []
    t1 = time()+60
    async with socket as tscm:
        while time() < t1:
            res = await tscm.recv()
            data.append(res['data'])
            data = trim_data(data)
            analyze_price_movement(data)
            '''            
            res_symbols = []
            for x in res:
                res_symbols.append((x['s'], str(x['E'])[7:]))
            print(res_symbols)
            '''
        df = pd.DataFrame(data)
        print(df.shape)
        df.to_csv('data', sep='\t')


def init_socket():
    coins, data_streams = [], []
    all_tickers = client.get_all_tickers()
    for ticker in all_tickers:
        if ticker['symbol'][-4:] == 'USDT':
            coins.append(ticker['symbol'][:-4])
            # data_streams.append(ticker['symbol'][:-4].lower() + 'usdt@miniTicker')
            # data_streams.append(ticker['symbol'][:-4].lower() + 'usdt@kline_1m')
            data_streams.append(ticker['symbol'][:-4].lower() + 'usdt@trade')
    # print(coins)
    print(data_streams)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(open_socket(data_streams[:1]))


def clear_balance():
    account_bals = client.get_account()['balances']
    print(account_bals)
    for coin in account_bals:
        ticker = coin['asset']+'BUSD'
        amount = float(coin['free'])
        if coin['asset'] not in invalid_quoteAsset and float(amount) > 0:
            print(coin, ticker)
            price = float(client.get_recent_trades(symbol=ticker)[-1]['price'])
            maxQty = float(client.get_symbol_info(ticker)['filters'][2]['maxQty'])
            if amount > maxQty:
                while amount >= maxQty:
                    client.order_market_sell(symbol=ticker, quantity=maxQty)
                    amount -= maxQty
            client.order_market_sell(symbol=ticker, quantity=amount)
    print(client.get_account()['balances'])


if __name__ == "__main__":
    # print(get_hist_data(start_date=str(time()-60*60*24), end_date=str(time())))
    # print(client.get_exchange_info()['symbols'])
    valid_tickers = {}
    invalid_quoteAsset = ['USDT', 'BUSD']
    for symbol in client.get_exchange_info()['symbols']:
        base = symbol['baseAsset']
        if symbol['baseAsset'] not in valid_tickers.keys():
            valid_tickers[base] = []
        valid_tickers[base].append(base+symbol['quoteAsset'])
    print(valid_tickers)
    clear_balance()
    client.order_market_buy(symbol='BTCBUSD', quoteOrderQty=1000)
    print(client.get_account()['balances'])

