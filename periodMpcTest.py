import pandas as pd
import numpy as np
import casadi
import matplotlib.pyplot as plt

from periodMpcPrev import PeriodMPC as PeriodMPCPrev
from periodEnumPrev import S as S_prev 
from periodEnumPrev import U as U_prev 
from periodMpc import PeriodMPC
from periodEnum import S, U
from plotResult import plotResult


# Closed-loop シミュレーション
refFile = "csv/genPath.csv"
speedRef= "csv/speed.csv"


df      = pd.read_csv(refFile)
refDist= df['Distance'].to_numpy()
refX   = df['x'].to_numpy()
refY   = df['y'].to_numpy()
cur    = df['Curvature'].to_numpy()
speed = df['Speed'].to_numpy()

df      = pd.read_csv(speedRef)
d       = df['d'].to_numpy()
vRef    = df['v'].to_numpy()
aRef    = df['a'].to_numpy()

speed  = casadi.interpolant('interp', 'linear', [d], vRef)
acc  = casadi.interpolant('interp', 'linear', [d], aRef)
curInterp= casadi.interpolant('interp', 'linear', [refDist], cur)
cur_diff= np.diff(cur)
curDiff= casadi.interpolant('interp', 'linear', [refDist[0:-1]], cur_diff)
aRefDiff = np.diff(aRef)
accDiff = casadi.interpolant('interp', 'linear', [d[0:-1]], aRefDiff)


N = 15

mpc = PeriodMPC(N,curInterp,accDiff,refDist[-1])
# mpc = PeriodMPC(N,curInterp,refDist[-1])
# mpc = PeriodMPC(N,cur,zhouDist[-1])

F = mpc.make_F()
mpc.make_nlp()

total   = mpc.nx*(mpc.N+1) + mpc.nu*mpc.N

x0      = casadi.DM.ones(total)

x       = casadi.DM.zeros(mpc.nx)
x[int(S.dDot)] = 10.0


xs      = [x]   # 状態
xx      = []    # 状態予測
us      = []    # 入力
t       = 0 
times   = [t]   # 時間（経路上の距離）

i = 0
sim_len = N
while t < 1000:
    u_opt,x0 = mpc.compute_optimal_control(x,x0)
    x = x0[len(S):len(S)*2:]
    # dDot = u_opt[0]
    dDot = x[int(S.dDot)]
    t = x[int(S.d)]
    xs.append(x)
    xx.append(x0)
    us.append(dDot)
    times.append(t)
    print('t =', t)
    print("u:  ",dDot)
    print("d:  ",x[int(S.d)])
    # print("var_cur:",x[int(S.var_cur)])
    print("------------------------")
    i += 1

# plt.plot(t, us)