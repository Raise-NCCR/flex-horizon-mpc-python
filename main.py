import pandas as pd
import numpy as np
import casadi
import time
import matplotlib.pyplot as plt

from vehicleMpc import VehicleMPC
from vehicleEnum import S, U
from plotResult import plotReuslt


# Closed-loop シミュレーション
# refFile = "csv/genPath.csv"
refFile = "csv/genPath.csv"

df      = pd.read_csv(refFile)
zhouDist= df['Distance'].to_numpy()
zhouX   = df['x'].to_numpy()
zhouY   = df['y'].to_numpy()
cur = df['Curvature'].to_numpy()

N = 15
sim = True

if sim:
    mpc = VehicleMPC(refFile, N)

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

    xs      = [x]   # 状態
    xx      = []    # 状態予測
    us      = []    # 入力
    t       = 0 
    times   = [t]   # 時間（経路上の距離）

    p_ts = [0]


    sim_len = zhouDist[-1]
    start = time.process_time()
    while t < 350 and x[int(S.dist)] < 5 and -5 < x[int(S.dist)]:
        step_start = time.process_time()
        dt,u_opt,x0 = mpc.compute_optimal_control(x,x0)
        step_end = time.process_time()

        step = 1.0
        ddt = int(dt/step)
        print("dt: ",dt)
        for i in range(ddt-1):
            x = F(x=x,u=u_opt,p=step)["x_next"]
            xs.append(x)
            xx.append(x0)
            us.append(u_opt)
            t = x[int(S.d)]
            times.append(t)
            p_ts.append(step_end-step_start)
            print('s= ', x[int(S.d)])
            print('t: ',x[int(S.t)])
            print("[x,y]: ",[x[int(S.x)], x[int(S.y)]])
            print("v: ", x[int(S.v)])
            print("psi: ", x[int(S.psi)])
            print("dist: ", x[int(S.dist)])
            print("------------------------")
        dt = dt - (ddt-1)*step
        if (dt != 0):
            x = F(x=x,u=u_opt,p=dt)["x_next"]
            # x = x0[len(S):len(S)*2:]
            t = x[int(S.d)]
            xs.append(x)
            xx.append(x0)
            us.append(u_opt)
            times.append(t)
            p_ts.append(step_end-step_start)
            print('s= ', x[int(S.d)])
            print('t: ',x[int(S.t)])
            print("[x,y]: ",[x[int(S.x)], x[int(S.y)]])
            print("v: ", x[int(S.v)])
            print("psi: ", x[int(S.psi)])
            print("dist: ", x[int(S.dist)])
            print("------------------------")

        # print('s= ', t)
        # print('t: ',x[int(S.t)])
        # print("u: ",u_opt)
        # print("[x,y]: ",[x[int(S.x)], x[int(S.y)]])
        # print("[refx,refy]: ",[mpc.refX(x[int(S.d)]), mpc.refY(x[int(S.d)])])
        # print("v: ",x[int(S.v)])
        # print("a: ", x[int(S.a)])
        # print("dist: ",x[int(S.dist)])
        # print("beta:    ",x[int(S.beta)])
        # print("------------------------")
    end = time.process_time()

    print('time', end-start)
    print('s= ', t)
    print('t: ',x[int(S.t)])
    print("[x,y]: ",[x[int(S.x)], x[int(S.y)]])
    print("------------------------")

    # xsD     = list(row[int(S.d)].full()[0][0] for row in xs)
    # plt.figure()
    # plt.plot(xsD, p_ts, '-')
    # plt.xlabel('t')
    # plt.ylabel('ds')
    # plt.grid()
    # plt.show()

    # plt.figure()
    # plt.plot(zhouDist, cur, '-')
    # plt.xlabel('t')
    # plt.ylabel('ds')
    # plt.grid()
    # plt.show()

    np.save('result/ts.npy', p_ts)
    np.save('result/us.npy', us)
    np.save('result/xs.npy', xs)
    np.save('result/xx.npy', xx)

us = np.load('result/us.npy')
xs = np.load('result/xs.npy')
xx = np.load('result/xx.npy')
show = [True] * (len(S)+len(U))
plotReuslt(xs, us, zhouX, zhouY, show)
