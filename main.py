import pandas as pd
import numpy as np
import casadi

from casadiMpc import MPC
from myEnum import S
from plotReuslt import plotReuslt


# Closed-loop シミュレーション
refFile = "csv/disCur.csv"

df      = pd.read_csv(refFile)
zhouDist= df['Distance'].to_numpy()
zhouX   = df['x'].to_numpy()
zhouY   = df['y'].to_numpy()

mpc = MPC(refFile)

ref             = np.zeros(mpc.nx)
ref[int(S.x)]   = zhouX[-1]
ref[int(S.y)]   = zhouY[-1]
mpc.set_ref(ref)

F = mpc.make_F()
mpc.make_nlp()

total   = mpc.nx*(mpc.N+1) + mpc.nu*mpc.N

x0      = casadi.DM.zeros(total)
x0[int(S.v):mpc.nx*(mpc.N+1):mpc.nx] = 16.7

x           = casadi.DM.zeros(mpc.nx)
x[int(S.v)] = 16.7

xs      = [x]
us      = []
t       = 0
times   = [t]

sim_len = zhouDist[-1]
while t < sim_len:
    u_opt,x0 = mpc.compute_optimal_control(x,x0)
    x = F(x=x,u=u_opt)["x_next"]
    t = x[int(S.d)]
    xs.append(x)
    us.append(u_opt)
    times.append(t)
    print('t =', t)
    print("u: ",u_opt)
    print("[x,y]: ",[x[int(S.x)], x[int(S.y)]])
    print("[refx,refy]: ",[mpc.refX(x[int(S.d)]), mpc.refY(x[int(S.d)])])
    print("v: ",x[int(S.v)])
    print("a: ", x[int(S.a)])
    print("dist: ",x[int(S.dist)])
    print("beta:    ",x[int(S.beta)])
    print("------------------------")

plotReuslt(xs, zhouX, zhouY)