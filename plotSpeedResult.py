import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from speedEnum import S, U
from rideComfort import rideComfort

# # シミュレーション結果をプロット
def plotSpeedResult(xs, us, refDist, refSpeed, show):
    xsD     = list(float(row[int(S.d)]) for row in xs)
    xsV     = list(float(row[int(S.v)]) for row in xs)
    xsA     = list(float(row[int(S.a)]) for row in xs)
    
    usA  = list(row[int(U.a)] for row in us)
    
    time = xsD
    
    output_df = pd.DataFrame({
        'd'     :xsD,
        'v'     :xsV,
        'a'     :xsA,
    })

    output_df.to_csv('csv/speed.csv', index=False)

    # output_df = pd.DataFrame({
    #     'a'     :usA,
    # })
    
    output_df.to_csv('result/mpcU.csv', index=False)
    num = 1

    # v
    if (show[int(S.v)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsV, '-')
        plt.plot(refDist, refSpeed, '-')
        plt.xlabel('t')
        plt.ylabel('v')
        plt.legend(['mpc','ref'])
        plt.grid()
        plt.show()
        num += 1

    # a
    if (show[int(S.a)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsA, '-')
        plt.xlabel('t')
        plt.ylabel('a')
        plt.grid()
        plt.show()
        num += 1

    # a
    if (show[len(S)+int(U.a)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time[1::], usA, '-')
        plt.xlabel('t')
        plt.ylabel('a')
        plt.grid()
        plt.show()
        num += 1