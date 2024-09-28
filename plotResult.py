import pandas as pd
import matplotlib.pyplot as plt

from myEnum import S

# # シミュレーション結果をプロット
def plotReuslt(xs, zhouX, zhouY):
    xsD     = list(row[int(S.d)].full()[0][0] for row in xs)
    xsV     = list(row[int(S.v)].full()[0][0] for row in xs)
    xsA     = list(row[int(S.a)].full()[0][0] for row in xs)
    xsBeta  = list(row[int(S.beta)].full()[0][0] for row in xs)
    xsomega = list(row[int(S.omega)].full()[0][0] for row in xs)
    xsPsi   = list(row[int(S.psi)].full()[0][0] for row in xs)
    xsX     = list(row[int(S.x)].full()[0][0] for row in xs)
    xsY     = list(row[int(S.y)].full()[0][0] for row in xs)
    xsAx    = list(row[int(S.ax)].full()[0][0] for row in xs)
    xsAy    = list(row[int(S.ay)].full()[0][0] for row in xs)
    xsXjerk = list(row[int(S.xJerk)].full()[0][0] for row in xs)

    output_df = pd.DataFrame({
        'v':    xsV,
        'a':    xsA,
        'beta': xsBeta,
        'omega':xsomega,
        'psi':  xsPsi,
        'x':    xsX,
        'y':    xsY,
        'ax':   xsAx,
        'ay':   xsAy,
        'xJerk':xsXjerk
    })

    output_df.to_csv('result/mpcPath.csv', index=False)

    plt.figure(1)
    plt.clf()
    plt.plot(xsX, xsY, '-')
    plt.plot(zhouX, zhouY, '-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['MPC','zhouPath'])
    plt.grid()
    plt.show()

    time = xsD
    plt.figure(2)
    plt.clf()
    plt.plot(time, xsAx, '-')
    plt.xlabel('t')
    plt.ylabel('ax')
    plt.grid()
    plt.show()

    plt.figure(3)
    plt.clf()
    plt.plot(time, xsAy, '-')
    plt.xlabel('t')
    plt.ylabel('ay')
    plt.grid()
    plt.show()

    plt.figure(4)
    plt.clf()
    plt.plot(time, xsXjerk, '-')
    plt.xlabel('t')
    plt.ylabel('xJerk')
    plt.grid()
    plt.show()

    plt.figure(5)
    plt.clf()
    plt.plot(time, xsV, '-')
    plt.xlabel('t')
    plt.ylabel('v')
    plt.grid()
    plt.show()