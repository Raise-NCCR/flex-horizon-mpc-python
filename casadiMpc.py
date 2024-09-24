from enum import IntEnum, auto
from math import pi
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.optimize import minimize
import casadi
from matplotlib import pyplot as plt

# state = v, a, b, wz, psi, x, y, dist, ax, ay, bDot, wzDot, xJerk, yJerk, i
# u = jerk, delta
# state の内容
class S(IntEnum):
    d       = 0
    v       = auto()
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
    dt      = auto()
class DS(IntEnum):
    dDot    = 0
    vDot    = auto()
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

class MPC:
    def __init__(self):
        self.car_l = 1.25
        self.a1 = -(32000.0+32000.0)/1100.0
        self.a2 = (32000.0*1.25-32000.0*1.15)/1100.0 - 1
        self.a3 = 32000.0/1100.0
        self.a4 = -1/1100.0
        self.b1 = (32000.0*1.25-32000.0*1.15)/1600
        self.b2 = -(32000.0*(1.15**2)+32000.0*((1.25 ** 2)))/1600
        self.b3 = (32000.0*1.15)/1600

        # 問題設定
        self.dt = 0.1  # 離散化ステップ
        self.ratio = 2
        self.N = 15     # ホライゾン離散化グリッド数 (MPCなので荒め)
        self.nx = len(S)      # 状態空間の次元
        self.ndx= len(DS)
        self.nu = len(U)      # 制御入力の次元

        # 重み係数
        q = np.zeros(self.nx)
        q[int(S.ax)] = 0.3
        q[int(S.ay)] = 0.3
        q[int(S.xJerk)] = 0.4
        self.Q = casadi.diag(q)
        s = np.ones(self.nx)
        self.S = casadi.diag(s)
        self.R = casadi.diag(np.ones(self.nu))

        # 経路の読み込み
        #path = pd.read_csv("newPath.csv")
        #self.distance = path['Distance'].to_numpy()
        #cur = path['Curvature'].to_numpy()
        #self.curvature = casadi.interpolant('interp', 'linear', [self.distance], cur)

        path = pd.read_csv("newPathC.csv")
        x = path['x'].to_numpy()
        y = path['y'].to_numpy()
        distance = path['Distance'].to_numpy()
        cur = path['Curvature'].to_numpy()
        self.cur = casadi.interpolant('interp', 'linear', [distance], cur)
        self.refX = casadi.interpolant('interp', 'linear', [distance], x)
        self.refY = casadi.interpolant('interp', 'linear', [distance], y)

        #self.pathRef = tmp.T.values.tolist()
        #self.kd_tree = KDTree([row[0:2] for row in self.pathRef])

        # 制約
        self.vmax = 16.7
        self.vmin = 0
        self.amax = 2
        self.amin = -1
        self.bmax = 10*pi/180.0
        self.bmin = -10*pi/180.0
        self.distmax = 0.65**2
        self.xJerkmax  = 1
        self.xJerkmin  = -1
        self.x_ub = [float('inf')] * self.nx
        self.x_ub[int(S.v)] = self.vmax
        self.x_ub[int(S.a)] = self.amax
        self.x_ub[int(S.b)] = self.bmax
        self.x_ub[int(S.dist)] = self.distmax
        self.x_ub[int(S.xJerk)] = self.xJerkmax
        self.x_lb = [-float('inf')] * self.nx
        self.x_lb[int(S.v)] = self.vmin
        self.x_lb[int(S.a)] = self.amin
        self.x_lb[int(S.b)] = self.bmin
        self.x_lb[int(S.xJerk)] = self.xJerkmin


        self.jerkmax = 1
        self.jerkmin = -1
        self.deltamax = 40*pi/180.0
        self.deltamin = -40*pi/180.0  
        self.u_ub = [self.jerkmax, self.deltamax]
        self.u_lb = [self.jerkmin, self.deltamin]

    def set_ref(self, ref):
        self.ref = ref

    def kinematics(self, state, control):
        v = state[int(S.v)]
        a = state[int(S.a)]
        b = state[int(S.b)]
        wz = state[int(S.wz)]
        psi = state[int(S.psi)]
        jerk = control[int(U.jerk)]
        delta = control[int(U.delta)]

        dDot = v
        vDot = a
        aDot = jerk
        bDot = casadi.atan(casadi.tan(delta)/2) - b
        wzDot = casadi.sin(b)*v/self.car_l - wz
        psiDot = wz
        xDot = v*casadi.cos(psi+b)
        yDot = v*casadi.sin(psi+b)
        ax = a
        ay = a*casadi.sin(b)
        xJerk = jerk
        yJerk = jerk*casadi.sin(b)
        return casadi.vertcat(dDot, vDot, aDot, bDot, wzDot, psiDot, xDot, yDot, ax, ay,xJerk, yJerk)

    def dynamics(self, state, control):
        v = state[int(S.v)]
        a = state[int(S.a)]
        b = state[int(S.b)]
        wz = state[int(S.wz)]
        psi = state[int(S.psi)]
        jerk = control[int(U.jerk)]
        delta = control[int(U.delta)]

        dDot = v
        vDot = a
        aDot = jerk
        bDot = self.a1*b/v + self.a2*wz/((v ** 2)) - wz + self.a3*delta/v + self.a4*a*b/v
        wzDot = self.b1*b + self.b2*wz/v + self.b3*delta
        psiDot = wz
        ax = a - v*b*wz
        ay = v*bDot + a*b + v*wz
        xDot = (casadi.cos(psi) * v - casadi.sin(psi) * v * b)
        yDot = (casadi.sin(psi) * v + casadi.cos(psi) * v * b)
        xJerk = jerk - a*b*wz - v*bDot*wz - v*b*wzDot
        yJerk = a*bDot + jerk*b + a*bDot + a*wz + v*wzDot
        return casadi.vertcat(dDot, vDot, aDot, bDot, wzDot, psiDot, xDot, yDot, ax, ay,xJerk, yJerk)

    # 状態更新関数
    def update_state(self, state, control):
        max_c = 0
        for i in range(30):
            c = casadi.fabs(self.cur(state[int(S.d)]+state[int(S.v)]*i*self.dt))
            max_c = casadi.if_else(c>max_c, c, max_c)
        dt = casadi.if_else(max_c>0.006, self.dt, self.dt*self.ratio)
        dstate = casadi.if_else(state[int(S.v)]>5, self.dynamics(state, control), self.kinematics(state, control))
        state_next = [
                    state[int(S.d)] + dstate[int(DS.dDot)] * dt,
                    state[int(S.v)] + dstate[int(DS.vDot)] * dt, 
                    state[int(S.a)] + dstate[int(DS.aDot)] * dt,
                    state[int(S.b)] + dstate[int(DS.bDot)] * dt,
                    state[int(S.wz)] + dstate[int(DS.wzDot)] * dt,
                    state[int(S.psi)] + dstate[int(DS.psiDot)] * dt,
                    state[int(S.x)] + dstate[int(DS.xDot)] * dt,
                    state[int(S.y)] + dstate[int(DS.yDot)] * dt,
                    0,
                    dstate[int(DS.bDot)],
                    dstate[int(DS.wzDot)],
                    dstate[int(DS.ax)],
                    dstate[int(DS.ay)],
                    dstate[int(DS.xJerk)],
                    dstate[int(DS.yJerk)],
                    dt,
                    ]
        d = state_next[int(S.d)]
        #state_next[int(S.dist)] = casadi.hypot(state_next[int(S.x)]-self.refX(d),state_next[int(S.y)]-self.refY(d))
        state_next[int(S.dist)] = (state_next[int(S.x)]-self.refX(d))**2+(state_next[int(S.y)]-self.refY(d))**2
        return casadi.vertcat(*state_next)

    def make_F(self):
        state = casadi.MX.sym('state', self.nx)
        control = casadi.MX.sym('control', self.nu)
        
        state_next = self.update_state(state, control)
        F = casadi.Function("F", [state, control],[state_next],["x","u"],["x_next"])
        return F

    def stage_cost(self, x, u):
        cost = casadi.dot(self.Q@x,x)
        #for i in range(self.N):
        #    dist = x[i][int(S.dist)]
        #    if dist > 0.65:
        #        cost = 10000000
        #    if dist < -0.65:
        #        cost = 10000000
        return cost
    
    def terminal_cost(self, x):
        diff = x - self.ref
        cost = casadi.dot(self.S@diff,diff)
        return cost
    
    def find_closest_point(self):
        dist, i = self.kd_tree.query([px, py], k=1)
        return dist, i

    def dist_ref(self,x,y):
        dist = casadi.sqrt((self.ref[int(S.x)]-x)**2 + (self.ref[int(S.y)]-y)**2)
        return dist

    def make_nlp(self):
        F = self.make_F()

        X = [casadi.MX.sym(f"x_{k}",self.nx) for k in range(self.N+1)]
        U = [casadi.MX.sym(f"u_{k}",self.nu) for k in range(self.N)]
        G = []

        J = 0
        for k in range(self.N):
            J += self.stage_cost(X[k],U[k])
            eq = X[k+1] - F(x=X[k],u=U[k])["x_next"]
            G.append(eq)
        J += self.terminal_cost(X[self.N])

        option = {"print_time":False,"ipopt":{"print_level":0}}
        nlp = {"x":casadi.vertcat(*X,*U),"f":J,"g":casadi.vertcat(*G)}
        self.S = casadi.nlpsol("S","ipopt",nlp,option)
        return

    def compute_optimal_control(self,x_init,x0):
        x_init = x_init.full().ravel().tolist()

        lbx = x_init + self.x_lb*self.N + self.u_lb*self.N 
        ubx = x_init + self.x_ub*self.N + self.u_ub*self.N
        lbg = [0]*self.nx*self.N 
        ubg = [0]*self.nx*self.N 

        res = self.S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
        offset = self.nx*(self.N+1)
        x0 = res["x"]
        u_opt = x0[offset:offset+self.nu]
        return u_opt, x0

# Closed-loop シミュレーション
sim_time = 7.0 # 10秒間のシミュレーション
mpc = MPC()
df = pd.read_csv('zhouPath.csv', header=None)
zhouX = df.iloc[0].values
zhouY = df.iloc[1].values
ref = np.zeros(mpc.nx)
ref[int(S.x)] = zhouX[-1]
ref[int(S.y)] = zhouY[-1]
mpc.set_ref(ref)
F = mpc.make_F()
mpc.make_nlp()
sim_steps = int(sim_time/mpc.dt)
total = mpc.nx*(mpc.N+1) + mpc.nu*mpc.N
x0 = casadi.DM.zeros(total)
x0[int(S.v):mpc.nx*(mpc.N+1):mpc.nx] = 16.7
state0 = casadi.DM.zeros(mpc.nx)
state0[int(S.v)] = 16.7
xs = [x0]
us = []
t = 0
times = [t]
x = state0
while t < sim_time:
    u_opt,x0 = mpc.compute_optimal_control(x,x0)
    x = F(x=x,u=u_opt)["x_next"]
    t = t + x[int(S.dt)]
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
    print("b:    ",x[int(S.b)])
    print("------------------------")


# # シミュレーション結果をプロット
# xs1 = [x[0] for x in xs]
# xs2 = [x[1] for x in xs]
# xs3 = [x[2] for x in xs]
# xs4 = [x[3] for x in xs]
# tgrid = [sampling_time*k for k in range(sim_steps)]

xsX = list(row[int(S.x)].full()[0][0] for row in xs)
xsY = list(row[int(S.y)].full()[0][0] for row in xs)
xsAx = list(row[int(S.ax)].full()[0][0] for row in xs)
xsAy = list(row[int(S.ay)].full()[0][0] for row in xs)
xsXjerk = list(row[int(S.xJerk)].full()[0][0] for row in xs)
output_df = pd.DataFrame({
    'x': xsX,
    'y': xsY,
    'ax': xsAx,
    'ay': xsAy,
    'xJerk': xsXjerk
})

output_df.to_csv('mpcPath.csv', index=False)

plt.figure(1)
plt.clf()
plt.plot(xsX, xsY, '-')
plt.plot(zhouX, zhouY, '-')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['MPC','zhouPath'])
plt.grid()
plt.show()

time = range(0,len(xsX))
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
