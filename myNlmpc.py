from enum import IntEnum, auto
from math import sin, cos
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.optimize import minimize

# state = v, a, b, wz, psi, x, y, dist, ax, ay, bDot, wzDot, xJerk, yJerk, i
# u = jerk, delta
# state の内容
class S(IntEnum):
    v       = 0
    a       = auto()
    b       = auto()
    wz      = auto()
    psi     = auto()
    x       = auto()
    y       = auto()
    dist    = auto()
    ax      = auto()
    ay      = auto()
    bDot    = auto()
    wzDot   = auto()
    xJerk   = auto()
    yJerk   = auto()
    i       = auto()
class DS(IntEnum):
    vDot    = 0
    aDot    = auto()
    bDot    = auto()
    wzDot   = auto()
    psiDot  = auto()
    xDot    = auto()
    yDot    = auto()
    ax      = auto()
    ay      = auto()
    xJerk   = auto()
    yJerk   = auto()
class U(IntEnum):
    jerk    = 0
    delta   = auto()

# Vehicle のダイナミクスを記述
class Vehicle:
    def __init__(self):
        self.a1 = -(32000.0+32000.0)/1100.0
        self.a2 = (32000.0*1.25-32000.0*1.15)/1100.0 - 1
        self.a3 = 32000.0/1100.0
        self.a4 = -1/1100.0
        self.b1 = (32000.0*1.25-32000.0*1.15)/1600
        self.b2 = -(32000.0*(1.15**2)+32000.0*((1.25 ** 2)))/1600
        self.b3 = (32000.0*1.15)/1600

    def dynamics(self, state, u):
        v = state[int(S.v)]
        a = state[int(S.a)]
        b = state[int(S.b)]
        wz = state[int(S.wz)]
        psi = state[int(S.psi)]
        jerk = u[int(U.jerk)]
        delta = u[int(U.delta)]
        vDot = a
        aDot = jerk
        bDot = self.a1*b/v + self.a2*wz/((v ** 2)) - wz + self.a3*delta/v + self.a4*a*b/v
        wzDot = self.b1*b + self.b2*wz/v + self.b3*delta
        psiDot = wz
        ax = a - v*b*wz
        ay = v*bDot + a*b + v*wz
        xDot = (cos(psi) * v - sin(psi) * v * b)
        yDot = (sin(psi) * v + cos(psi) * v * b)
        xJerk = jerk - a*b*wz - v*bDot*wz - v*b*wzDot
        yJerk = a*bDot + jerk*b + a*bDot + a*wz + v*wzDot
        return vDot, aDot, bDot, wzDot, psiDot, xDot, yDot, ax, ay, xJerk, yJerk



class MPC:
    def __init__(self):
        # Vehicle のダイナミクス
        self.vehicle = Vehicle()
        # 問題設定
        self.dt = 0.1  # 離散化ステップ
        self.N = 10      # ホライゾン離散化グリッド数 (MPCなので荒め)
        self.nx = len(S)      # 状態空間の次元
        self.nu = len(U)      # 制御入力の次元

        # 定数
        self.q1 = 0.5
        self.q2 = 0.2
        self.q3 = 0.15
        self.q4 = 0.15

        # 経路の読み込み
        tmp = pd.read_csv("zhouPath.csv", header=None)
        self.pathRef = tmp.T.values.tolist()
        self.kd_tree = KDTree([row[0:2] for row in self.pathRef])

        # 制約
        self.bounds = [(-1, 1), (-1.57, 1.57)] * self.N


    # 状態更新関数
    def update_state(self, state, u):
        dt = self.dt
        dstate = self.vehicle.dynamics(state, u)
        dist, i = self.find_closest_point(state[int(S.x)] + dstate[int(DS.xDot)] * dt, state[int(S.y)] + dstate[int(DS.yDot)] * dt)
        state_next =   [state[int(S.v)] + dstate[int(DS.vDot)] * dt, 
                    state[int(S.a)] + dstate[int(DS.aDot)] * dt,
                    state[int(S.b)] + dstate[int(DS.bDot)] * dt,
                    state[int(S.wz)] + dstate[int(DS.wzDot)] * dt,
                    state[int(S.psi)] + dstate[int(DS.psiDot)] * dt,
                    state[int(S.x)] + dstate[int(DS.xDot)] * dt,
                    state[int(S.y)] + dstate[int(DS.yDot)] * dt,
                    dist,
                    dstate[int(DS.bDot)],
                    dstate[int(DS.wzDot)],
                    dstate[int(DS.ax)],
                    dstate[int(DS.ay)],
                    dstate[int(DS.xJerk)],
                    dstate[int(DS.yJerk)],
                    i]
        return state_next

    def solve(self, state, u_init=None):
        # 初期状態
        if u_init == None:
            u_init = np.zeros(self.nu*self.N)
        cons = [
                {'type': 'ineq', 'fun': self.lb_dist_cons, 'args': state},
                {'type': 'ineq', 'fun': self.ub_dist_cons, 'args': state},
                {'type': 'ineq', 'fun': self.lb_v_cons, 'args': state},
                {'type': 'ineq', 'fun': self.ub_v_cons, 'args': state},
                {'type': 'ineq', 'fun': self.lb_ax_cons, 'args': state},
                {'type': 'ineq', 'fun': self.ub_ax_cons, 'args': state},
                {'type': 'ineq', 'fun': self.lb_xJerk_cons, 'args': state},
                {'type': 'ineq', 'fun': self.ub_xJerk_cons, 'args': state},
            ]
        result = minimize(self.cost_function, u_init, args=state, bounds=self.bounds, constraints=cons, method='SLSQP')
        return result.x[0:2] # 制御入力を return

    def cost_function(self, u, *args):
        state = args[0]
        cost = 0
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
            v = state[int(S.v)]
            dist = state[int(S.dist)]
            ax = state[int(S.ax)]
            ay = state[int(S.ay)]
            xJerk = state[int(S.xJerk)]
            # cost += (pow(self.q1, 2)*pow((16.7-v), 2)) + (pow(self.q2, 2) * pow(xJerk, 2)) + (pow(self.q3, 2) * pow(ax, 2)) + (pow(self.q4, 2) * pow(ay, 2))
            cost += (self.q2 * (xJerk ** 2)) + (self.q3 * (ax ** 2)) + (self.q4 * (ay ** 2))
            if v > 17:
                cost = 10000
            if dist > 0.65:
                cost = 10000
            if dist < -0.65:
                cost = 10000
        return cost
    
    def find_closest_point(self, px, py):
        dist, i = self.kd_tree.query([px, py], k=1)
        return dist, i
    
    def lb_dist_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return state[int(S.dist)] + 0.65
    
    def ub_dist_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return 0.65 - state[int(S.dist)]
    
    def lb_v_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return state[int(S.v)] - 5

    def ub_v_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return 16.7 - state[int(S.v)]
    
    def lb_ax_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return state[int(S.ax)] + 1

    def ub_ax_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return 2 - state[int(S.ax)]
    
    def lb_xJerk_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return state[int(S.xJerk)] + 1

    def ub_xJerk_cons(self, u, *args):
        state = args
        for i in range(self.N):
            state = self.update_state(state, u[i*2:(i+1)*2])
        return 1 - state[int(S.xJerk)]


# Closed-loop シミュレーション
sim_time = 6.0 # 10秒間のシミュレーション
mpc = MPC()
sim_steps = int(sim_time/mpc.dt)
state0 = np.zeros(mpc.nx)
state0[int(S.v)] = 16.7
u0 = np.zeros(mpc.nu)
states = [state0]
us = [u0]
state = state0
for step in range(sim_steps):
    print('t =', step*mpc.dt)
    u = mpc.solve(state)
    state = mpc.update_state(state, u)
    print(state)
    print(u)
    states.append(state)
    us.append(u)


# # シミュレーション結果をプロット
# xs1 = [x[0] for x in xs]
# xs2 = [x[1] for x in xs]
# xs3 = [x[2] for x in xs]
# xs4 = [x[3] for x in xs]
# tgrid = [sampling_time*k for k in range(sim_steps)]

# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.clf()
# plt.plot(tgrid, xs1, '--')
# plt.plot(tgrid, xs2, '-')
# plt.plot(tgrid, xs3, '-')
# plt.plot(tgrid, xs4, '-')
# plt.step(tgrid, us, '-.')
# plt.xlabel('t')
# plt.legend(['y(x1)','th(x2)', 'dy(x3)', 'dth(x4)','u'])
# plt.grid()
# plt.show()