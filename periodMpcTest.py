import pandas as pd
import numpy as np
import casadi

from periodMpc import PeriodMPC
from periodEnum import S, U
from plotResult import plotReuslt


# Closed-loop シミュレーション
refFile = "csv/disCur.csv"

df      = pd.read_csv(refFile)
zhouDist= df['Distance'].to_numpy()
zhouX   = df['x'].to_numpy()
zhouY   = df['y'].to_numpy()
cur     = df['Curvature'].to_numpy()

cur    = casadi.interpolant('interp', 'linear', [zhouDist], cur)

# cur_diff= np.diff(cur)
# curDiff= casadi.interpolant('interp', 'linear', [zhouDist[:len(zhouDist)-1:]], cur_diff)

N = 10

# mpc = PeriodMPC(N,curDiff,zhouDist[-1])
mpc = PeriodMPC(N,cur,zhouDist[-1])

F = mpc.make_F()
mpc.make_nlp()

total   = mpc.nx*(mpc.N+1) + mpc.nu*mpc.N

x0      = casadi.DM.ones(total)

x       = casadi.DM.zeros(mpc.nx)

xs      = [x]   # 状態
xx      = []    # 状態予測
us      = []    # 入力
t       = 0 
times   = [t]   # 時間（経路上の距離）

i = 0
sim_len = N
while i < sim_len:
    u_opt,x0 = mpc.compute_optimal_control(x,x0)
    x = x0[len(S):len(S)*2:]
    t = x[int(S.d)]
    xs.append(x)
    xx.append(x0)
    us.append(u_opt)
    times.append(t)
    print('t =', t)
    print("u:  ",u_opt)
    print("d:  ",x[int(S.d)])
    print("var:",x[int(S.var)])
    print("------------------------")
    i += 1

print(xx)