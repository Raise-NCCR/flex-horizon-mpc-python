import numpy as np
import pandas as pd

from vehicleEnum import S, U
from plotResult import plotReuslt

refFile = "csv/genPath.csv"

df      = pd.read_csv(refFile)
zhouDist= df['Distance'].to_numpy()
zhouX   = df['x'].to_numpy()
zhouY   = df['y'].to_numpy()
cur = df['Curvature'].to_numpy()

xs = np.load('result/xs.npy')
us = np.load('result/us.npy')
ts = np.load('result/ts.npy')
xx = np.load('result/xx.npy')

show = [True] * (len(S)+len(U))
plotReuslt(xs, us, zhouX, zhouY, show)
