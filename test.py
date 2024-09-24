import numpy as np
import pandas as pd
import casadi as ca

df = pd.read_csv('newPathC.csv')
dis = df['Distance'].to_numpy()
rho = df['Curvature'].to_numpy()
x = df['x'].to_numpy()
y = df['y'].to_numpy()
x = ca.interpolant('interp', 'linear', [dis], x)
y = ca.interpolant('interp', 'linear', [dis], y)
print(x(200), y(200))

#df = pd.read_csv('zhouPath.csv', header=None)
#dfX = df.iloc[0].values
#dfY = df.iloc[1].values
#print(dfX[-1], dfY[-1])
