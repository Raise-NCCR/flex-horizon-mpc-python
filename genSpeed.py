import pandas as pd
import numpy as np
import casadi
import time
import matplotlib.pyplot as plt

from vehicleMpc import VehicleMPC
from vehicleEnum import S, U
from plotSpeedResult import plotSpeedResult
from execPeriodMpc import execPeriodMpc
from speedMpc import SpeedMPC


# Closed-loop シミュレーション
# refFile = "csv/genPath.csv"
refFile = "csv/genPath.csv"

df      = pd.read_csv(refFile)
refDist= df['Distance'].to_numpy()
refX   = df['x'].to_numpy()
refY   = df['y'].to_numpy()
cur = df['Curvature'].to_numpy()
speed = df['Speed'].to_numpy()


N = 100
sim = True

if sim:
    speedMPC = SpeedMPC(refFile, N)

    F = speedMPC.make_F()
    speedMPC.make_nlp()

    total   = speedMPC.nx*(speedMPC.N+1) + speedMPC.nu*speedMPC.N

    x0      = casadi.DM.zeros(total)
    x0[int(S.v):speedMPC.nx*(speedMPC.N+1):speedMPC.nx] = 16.7

    x           = casadi.DM.zeros(speedMPC.nx)
    x[int(S.v)] = 60/3.6

    xs      = [x]   # 状態
    xx      = []    # 状態予測
    us      = []    # 入力
    t       = 0 
    times   = [t]   # 時間（経路上の距離）

    p_ts = [0]


    sim_len = refDist[-1]
    start = time.process_time()
    while x[int(S.d)] < 1000:
        dt = np.ones(speedMPC.N)
        u, vRef, x0 = speedMPC.compute_optimal_control(x,x0,dt)
        x = F(x=x, u=u, p=dt[0])['x_next']
        # print('d: ', x[int(S.d)])
        # print('v: ', x[int(S.v)])
        # print('a: ', x[int(S.a)])
        # print('u: ', u)
        xs.append(x)
        us.append(u)


        
    np.save('result/ts.npy', p_ts)
    np.save('result/us.npy', us)
    np.save('result/xs.npy', xs)
    np.save('result/xx.npy', xx)

us = np.load('result/us.npy')
xs = np.load('result/xs.npy')
xx = np.load('result/xx.npy')
show = [True] * (len(S)+len(U))
plotSpeedResult(xs, us, refDist, speed, show)
