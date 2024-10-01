import pandas as pd
import numpy as np
import casadi

from vehicleEnum import S as VS

from periodMpc import PeriodMPC
from periodEnum import S, U
from plotResult import plotReuslt


# Closed-loop シミュレーション
def execPeriodMpc(curDiff, cur_x0, N, dest):
    mpc = PeriodMPC(N, curDiff, dest)

    F = mpc.make_F()
    mpc.make_nlp()

    total   = mpc.nx*(mpc.N+1) + mpc.nu*mpc.N

    x0      = casadi.DM.ones(total)
    x0[int(S.d)] = cur_x0[int(VS.d)]

    x       = casadi.DM.zeros(mpc.nx)
    x[int(S.d)] = cur_x0[int(VS.d)]

    us      = []    # 入力

    i = 0
    sim_len = N
    while i < sim_len:
        u_opt,x0 = mpc.compute_optimal_control(x,x0)
        x = F(x=x,u=u_opt)["x_next"]
        us.append(u_opt)
        i += 1

    return casadi.vertcat(*us)