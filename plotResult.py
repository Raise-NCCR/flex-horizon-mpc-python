import pandas as pd
import matplotlib.pyplot as plt

from vehicleEnum import S, U

# # シミュレーション結果をプロット
def plotReuslt(xs, us, zhouX, zhouY, show):
    xsD     = list(row[int(S.d)].full()[0][0] for row in xs)
    xsV     = list(row[int(S.v)].full()[0][0] for row in xs)
    xsA     = list(row[int(S.a)].full()[0][0] for row in xs)
    xsBeta  = list(row[int(S.beta)].full()[0][0] for row in xs)
    xsDelta = list(row[int(S.delta)].full()[0][0] for row in xs)
    xsOmega = list(row[int(S.omega)].full()[0][0] for row in xs)
    xsPsi   = list(row[int(S.psi)].full()[0][0] for row in xs)
    xsX     = list(row[int(S.x)].full()[0][0] for row in xs)
    xsY     = list(row[int(S.y)].full()[0][0] for row in xs)
    xsAx    = list(row[int(S.ax)].full()[0][0] for row in xs)
    xsAy    = list(row[int(S.ay)].full()[0][0] for row in xs)
    xsXjerk = list(row[int(S.xJerk)].full()[0][0] for row in xs)

    usJerk  = list(row[int(U.jerk)].full()[0][0] for row in us)
    usDelta = list(row[int(U.deltaDot)].full()[0][0] for row in us)

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
    plt.plot(zhouX, zhouY, '-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['MPC','zhouPath'])
    plt.grid()
    plt.show()
    num += 1

    time = xsD
    # v
    if (show[int(S.v)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsV, '-')
        plt.xlabel('t')
        plt.ylabel('v')
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

    # beta
    if (show[int(S.beta)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsBeta, '-')
        plt.xlabel('t')
        plt.ylabel('beta')
        plt.grid()
        plt.show()
        num += 1

    # delta
    if (show[int(S.delta)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsDelta, '-')
        plt.xlabel('t')
        plt.ylabel('delta')
        plt.grid()
        plt.show()
        num += 1

    # omega
    if (show[int(S.omega)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsOmega, '-')
        plt.xlabel('t')
        plt.ylabel('omega')
        plt.grid()
        plt.show()
        num += 1

    # psi
    if (show[int(S.psi)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsPsi, '-')
        plt.xlabel('t')
        plt.ylabel('psi')
        plt.grid()
        plt.show()
        num += 1

    # ax
    if (show[int(S.ax)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsAx, '-')
        plt.xlabel('t')
        plt.ylabel('ax')
        plt.grid()
        plt.show()
        num += 1

    # ay
    if (show[int(S.ay)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsAy, '-')
        plt.xlabel('t')
        plt.ylabel('ay')
        plt.grid()
        plt.show()
        num += 1

    # xJerk
    if (show[int(S.xJerk)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time, xsXjerk, '-')
        plt.xlabel('t')
        plt.ylabel('xJerk')
        plt.grid()
        plt.show()
        num += 1

    # jerk
    if (show[len(S)+int(U.jerk)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time[1::], usJerk, '-')
        plt.xlabel('t')
        plt.ylabel('jerk')
        plt.grid()
        plt.show()
        num += 1

    # delta
    if (show[len(S)+int(U.deltaDot)]):
        plt.figure(num)
        plt.clf()
        plt.plot(time[1::], usDelta, '-')
        plt.xlabel('t')
        plt.ylabel('delta')
        plt.grid()
        plt.show()
        num += 1