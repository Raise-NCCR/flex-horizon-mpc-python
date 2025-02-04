import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from vehicleEnum import S, U
from rideComfort import rideComfort

# # シミュレーション結果をプロット
def plotReuslt(xs, us, refX, refY, show):
    xsD     = list(row[int(S.d)] for row in xs)
    xsV     = list(row[int(S.v)] for row in xs)
    xsA     = list(row[int(S.a)] for row in xs)
    xsBeta  = list(row[int(S.beta)] for row in xs)
    xsDelta = list(row[int(S.delta)] for row in xs)
    xsOmega = list(row[int(S.omega)] for row in xs)
    xsPsi   = list(row[int(S.psi)] for row in xs)
    xsTheta   = list(row[int(S.theta)] for row in xs)
    xsX     = list(row[int(S.x)] for row in xs)
    xsY     = list(row[int(S.y)] for row in xs)
    xsDist     = list(row[int(S.dist)] for row in xs)
    xsAx    = list(row[int(S.ax)] for row in xs)
    xsAy    = list(row[int(S.ay)] for row in xs)
    xsXjerk = list(row[int(S.xJerk)] for row in xs)
    xsYjerk = list(row[int(S.yJerk)] for row in xs)

    usJerk  = list(row[int(U.jerk)] for row in us)
    usDelta = list(row[int(U.deltaDot)] for row in us)

    xsComfort = rideComfort(xs)

    xs = np.load('result/default/xs.npy')
    us = np.load('result/default/us.npy')

    def_xsD     = list(row[int(S.d)] for row in xs)
    def_xsV     = list(row[int(S.v)] for row in xs)
    def_xsA     = list(row[int(S.a)] for row in xs)
    def_xsBeta  = list(row[int(S.beta)] for row in xs)
    def_xsDelta = list(row[int(S.delta)] for row in xs)
    def_xsOmega = list(row[int(S.omega)] for row in xs)
    def_xsPsi   = list(row[int(S.psi)] for row in xs)
    def_xsTheta   = list(row[int(S.theta)] for row in xs)
    def_xsX     = list(row[int(S.x)] for row in xs)
    def_xsY     = list(row[int(S.y)] for row in xs)
    def_xsDist     = list(row[int(S.dist)] for row in xs)
    def_xsAx    = list(row[int(S.ax)] for row in xs)
    def_xsAy    = list(row[int(S.ay)] for row in xs)
    def_xsXjerk = list(row[int(S.xJerk)] for row in xs)
    def_xsYjerk = list(row[int(S.yJerk)] for row in xs)

    def_usJerk  = list(row[int(U.jerk)] for row in us)
    def_usDelta = list(row[int(U.deltaDot)] for row in us)

    def_xsComfort = rideComfort(xs)
    
    ts = np.load('result/ts.npy')
    def_ts = np.load('result/default/ts.npy')

    time = xsD
    def_time = def_xsD

    output_df = pd.DataFrame({
        'v'     :xsV,
        'a'     :xsA,
        'beta'  :xsBeta,
        'delta' :xsDelta,
        'omega' :xsOmega,
        'psi'   :xsPsi,
        'x'     :xsX,
        'y'     :xsY,
        'ax'    :xsAx,
        'ay'    :xsAy,
        'xJerk' :xsXjerk,
    })

    output_df.to_csv('result/mpcX.csv', index=False)

    output_df = pd.DataFrame({
        'jerk'  :usJerk,
        'delta' :usDelta,
    })
    
    output_df.to_csv('result/mpcU.csv', index=False)
    num = 1

    plt.figure(num)
    plt.clf()
    plt.plot(xsX, xsY, '-')
    plt.plot(refX, refY, '-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['MPC','RefPath'])
    plt.grid()
    plt.show()
    num += 1

    plt.figure(num)
    plt.clf()
    plt.plot(time, ts, '-')
    plt.plot(def_time, def_ts, '-')
    plt.xlabel('t')
    plt.ylabel('compute time')
    plt.legend(['flex','fixed'])
    plt.grid()
    plt.show()
    num += 1

    
    plt.figure(num)
    plt.clf()
    plt.plot(time, xsComfort, '-')
    plt.plot(def_time, def_xsComfort, '-')
    plt.xlabel('t')
    plt.ylabel('ride comfort')
    plt.legend(['flex','fixed'])
    plt.grid()
    plt.show()
    num += 1

    # v
    if (show[int(S.v)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsV, '-')
        plt.plot(def_time, def_xsV, '-')
        plt.xlabel('t')
        plt.ylabel('v')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # a
    if (show[int(S.a)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsA, '-')
        plt.plot(def_time, def_xsA, '-')
        plt.xlabel('t')
        plt.ylabel('a')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # beta
    if (show[int(S.beta)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsBeta, '-')
        plt.plot(def_time, def_xsBeta, '-')
        plt.xlabel('t')
        plt.ylabel('beta')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # delta
    if (show[int(S.delta)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsDelta, '-')
        plt.plot(def_time, def_xsDelta, '-')
        plt.xlabel('t')
        plt.ylabel('delta')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # omega
    if (show[int(S.omega)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsOmega, '-')
        plt.plot(def_time, def_xsOmega, '-')
        plt.xlabel('t')
        plt.ylabel('omega')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # dist
    if (show[int(S.dist)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsDist, '-')
        plt.plot(def_time, def_xsDist, '-')
        plt.xlabel('t')
        plt.ylabel('dist')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # psi
    if (show[int(S.psi)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsPsi, '-')
        plt.plot(def_time, def_xsPsi, '-')
        plt.xlabel('t')
        plt.ylabel('psi')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # theta
    if (show[int(S.theta)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsTheta, '-')
        plt.plot(def_time, def_xsTheta, '-')
        plt.xlabel('t')
        plt.ylabel('theta')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # ax
    if (show[int(S.ax)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsAx, '-')
        plt.plot(def_time, def_xsAx, '-')
        plt.xlabel('t')
        plt.ylabel('ax')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # ay
    if (show[int(S.ay)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsAy, '-')
        plt.plot(def_time, def_xsAy, '-')
        plt.xlabel('t')
        plt.ylabel('ay')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # xJerk
    if (show[int(S.xJerk)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsXjerk, '-')
        plt.plot(def_time, def_xsXjerk, '-')
        plt.xlabel('t')
        plt.ylabel('xJerk')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # yJerk
    if (show[int(S.yJerk)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsYjerk, '-')
        plt.plot(def_time, def_xsYjerk, '-')
        plt.xlabel('t')
        plt.ylabel('yJerk')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # jerk
    if (show[len(S)+int(U.jerk)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time[1::], usJerk, '-')
        plt.plot(def_time[1::], def_usJerk, '-')
        plt.xlabel('t')
        plt.ylabel('jerk')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1

    # delta
    if (show[len(S)+int(U.deltaDot)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time[1::], usDelta, '-')
        plt.plot(def_time[1::], def_usDelta, '-')
        plt.xlabel('t')
        plt.ylabel('deltaDot')
        plt.legend(['flex','fixed'])
        plt.grid()
        plt.show()
        num += 1