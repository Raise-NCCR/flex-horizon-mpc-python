import pandas as pd
import numpy as np
import casadi
import time
import math
import matplotlib.pyplot as plt

from vehicleMpc import VehicleMPC
from vehicleEnum import S, U
from plotResult import plotResult
from execPeriodMpc import execPeriodMpc
from speedMpc import SpeedMPC


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

speed  = casadi.interpolant('interp', 'linear', [d], vRef)
curInterp= casadi.interpolant('interp', 'linear', [refDist], cur)


N = 15
sim = True

if sim:
    mpc = VehicleMPC(refFile, N, speed)
    
    ref             = np.zeros(mpc.nx)
    ref[int(S.x)]   = refX[-1]
    ref[int(S.y)]   = refY[-1]
    
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
    coorErr = [0]
    speedErr= [0]

    p_ts = [0]


    sim_len = refDist[-1]
    start = time.process_time()
    while t < 1000 and x[int(S.dist)] < 5 and -5 < x[int(S.dist)]:
        step_start = time.process_time()
        dt = execPeriodMpc(curInterp, x0, N, refDist[-1])
        dt = dt.full().ravel().tolist()
        dt = list(map(lambda y: 1.0 if y < 1.0 else y, dt))
        dt = list(map(lambda y: int(y*10)*0.1, dt))
        dt,u_opt,x0 = mpc.compute_optimal_control(x,x0,dt)
        step_end = time.process_time()

        step = 1.0
        ddt = int(dt/step)
        print("dt: ",dt)
        for i in range(ddt-1):
            x = F(x=x,u=u_opt,dt=step)["x_next"]
            xs.append(x)
            xx.append(x0)
            us.append(u_opt)
            p_ts.append(step_end-step_start)
            print('s= ', x[int(S.d)])
            print('t: ',x[int(S.t)])
            print("[x,y]: ",[x[int(S.x)], x[int(S.y)]])
            print("v: ", x[int(S.v)])
            print("theta: ", x[int(S.theta)])
            print("dist: ", x[int(S.dist)])
            print("------------------------")
        dt = dt - (ddt-1)*step
        if (dt != 0):
            x = F(x=x,u=u_opt,dt=dt)["x_next"]
            # x = x0[len(S):len(S)*2:]
            xs.append(x)
            xx.append(x0)
            us.append(u_opt)
            p_ts.append(step_end-step_start)
            print('s= ', x[int(S.d)])
            print('t: ',x[int(S.t)])
            print("[x,y]: ",[x[int(S.x)], x[int(S.y)]])
            print("v: ", x[int(S.v)])
            print("theta: ", x[int(S.theta)])
            print("dist: ", x[int(S.dist)])
            print("------------------------")
        xPred = x0[mpc.nx:mpc.nx*2]
        speedPred = xPred[int(S.v)].full()[0][0]
        coorPredX = xPred[int(S.x)].full()[0][0]
        coorPredY = xPred[int(S.y)].full()[0][0]
        coorX = x[int(S.x)].full()[0][0]
        coorY = x[int(S.y)].full()[0][0]
        speedErr.append(speedPred-x[int(S.v)].full()[0][0])
        coorErr.append(math.sqrt((coorPredX-coorX)**2+(coorPredY-coorY)**2))
        t = x[int(S.d)].full()[0][0]
        times.append(t)

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

    output_df = pd.DataFrame({
        'sppedErr'  :speedErr,
        'coorErr'   :coorErr,
    })

    output_df.to_csv('result/err.csv', index=False)

    xsD     = list(row[int(S.d)].full()[0][0] for row in xs)

    plt.figure()
    plt.plot(times, speedErr, '-')
    plt.xlabel('t[m]')
    plt.ylabel('Speed prediction error[m/s]')
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(times, coorErr, '-')
    plt.xlabel('t[m]')
    plt.ylabel('Coordinate prediction error[m]')
    plt.grid()
    plt.show()

    np.save('result/ts.npy', p_ts)
    np.save('result/us.npy', us)
    np.save('result/xs.npy', xs)
    np.save('result/xx.npy', xx)

us = np.load('result/us.npy')
xs = np.load('result/xs.npy')
xx = np.load('result/xx.npy')
show = [True] * (len(S)+len(U))
plotResult(xs, us, refX, refY, show)
