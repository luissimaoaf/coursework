import yfinance as yf

sp500 = yf.download('^GSPC', start='2020-02-01')
nasdaq = yf.download('^IXIC', start='2020-02-01')
dowjones = yf.download('^DJI', start='2020-02-01')
ftse = yf.download('^FTSE', start='2020-02-01')
nikkei = yf.download('^N225', start='2020-02-01')
stoxx = yf.download('^STOXX', start='2020-02-01')


sp500.to_csv('sp500.csv')
nasdaq.to_csv('nasdaq.csv')
dowjones.to_csv('dowjones.csv')
ftse.to_csv('ftse.csv')
nikkei.to_csv('nikkei.csv')
stoxx.to_csv('stoxx.csv')
