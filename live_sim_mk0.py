import pandas as pd
from binance.client import Client
from binance import AsyncClient, BinanceSocketManager
import asyncio
from time import time
from scipy.signal import savgol_filter
import ta

api_key = "5zt3nLqIbfc4b2g0iT1aiI0rxWrD16KNNETF3Fc3xyVDORP0s4J44bcObkNQHgKN"
api_secret = "7G6wg0NhGtrSIGtX8FXEAlgia3I6pFVPj6sa52nvUdgaV6tCyuHJ3OZkZBb1N08R"
client = Client(api_key, api_secret, testnet=True)


def get_hist_data(ticker='BTCUSDT', interval='1m', start_date='1 Jan,2021', end_date='31 Dec, 2021'):
    data = pd.DataFrame(client.get_historical_klines(ticker, interval, start_date, end_date))
    data.columns = ['open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time', 'qav', 'num_trades',
                    'taker_base_vol', 'taker_quote_vol', 'ignore']
    data = data[['close_time', 'Close']].copy()
    return data.values.tolist()


def parse_data(data):
    data['SMA 7'] = ta.trend.sma_indicator(data['c'], window=7)
    data['sav gol'] = savgol_filter(data['c'], 25, 3)

    sav_gol = data['sav gol'].shift(1)
    p7 = data['SMA 7'].shift(1)

    data['Pos Crossing'] = (data['sav gol'] >= data['SMA 7']) & (sav_gol <= p7)
    data['Neg Crossing'] = (data['sav gol'] <= data['SMA 7']) & (sav_gol >= p7)

    return data


async def open_socket(ticker, interval='1m'):
    wallet = {'USDT': 10000, ticker: 0}
    fees = 0.9985

    bm_client = await AsyncClient.create()
    bm = BinanceSocketManager(bm_client)
    socket = bm.kline_socket(ticker, interval=interval)
    raw_data = []
    closed_data = get_hist_data(start_date=str(time()-60*60*24), end_date=str(time()))
    async with socket as tscm:
        while True:
            res = await tscm.recv()
            raw_data.append(res)
            if res['k']['x']:
                res['k']['c'] = float(res['k']['c'])
                closed_data.append([res['k']['T'], res['k']['c']])

                data = pd.DataFrame(closed_data)
                data.columns = ['T', 'c']
                data = parse_data(data=data)
                data.to_csv('live data ' + res['s'] + '.csv', index=False, header=True)

                # buy / sell
                if data.iloc[-1]['Pos Crossing']:
                    print('BUY', data.iloc[-1])
                    if wallet['USDT'] > 0:
                        wallet[ticker] = (wallet['USDT'] / data.iloc[-1]['c']) * fees
                        wallet['USDT'] = 0
                elif data.iloc[-1]['Neg Crossing']:
                    print('SELL', data.iloc[-1])
                    if wallet[ticker] > 0:
                        wallet['USDT'] = wallet[ticker] * data.iloc[-1]['c'] * fees
                        wallet[ticker] = 0
                print(wallet)


def init_socket():
    ticker = 'BTCBUSD'
    loop = asyncio.get_event_loop()
    loop.run_until_complete(open_socket(ticker, interval='1m'))


if __name__ == "__main__":
    init_socket()

