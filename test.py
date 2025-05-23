import pandas as pd
import numpy as np
import casadi as ca
from scipy import interpolate

from vehicleEnum import S, U

# refFile = "csv/disCur.csv"

# df      = pd.read_csv(refFile)
# zhouDist= df['Distance'].to_numpy()
# zhouX   = df['x'].to_numpy()
# zhouY   = df['y'].to_numpy()
# cur     = df['Curvature'].to_numpy()

# curDiff = np.diff(cur)

# # cur = interpolate.interp1d(zhouDist[0:len(zhouDist)-1], curDiff)
# cur = interpolate.interp1d(zhouDist, cur)

# ds = np.linspace(0, 100, 100)
# cur = cur(ds)
# print(cur)
# mean_cur = np.mean(cur)
# diff = cur - mean_cur
# var = np.dot(diff, diff)/100
# print(var)
# print(xx[1][int(S.x)], xx[1][int(S.y)])

n = ca.MX([0])
print(ca.sqrt(n))