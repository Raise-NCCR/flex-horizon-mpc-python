import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from vehicleEnum import S, U
from plotResult import plotReuslt

x = np.load('result/xs.npy')
u = np.load('result/us.npy')
df = pd.read_csv('csv/disCur.csv')
refX = df['x'].to_numpy()
refY = df['y'].to_numpy()
s = [True] * len(S)
plotReuslt(x,u,refX,refY,s)
