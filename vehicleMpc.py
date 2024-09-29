from math import pi
import numpy as np
import pandas as pd
import casadi
from matplotlib import pyplot as plt

from myEnum     import S, DS, U
from vehicle    import Vehicle

class MPC:
    def __init__(self, refFile):
        # 問題設定
        self.dt     = 1.5       # 離散化ステップ
        self.ratio  = 2
        self.N      = 30        # ホライゾン離散化グリッド数 (MPCなので荒め)
        self.nx     = len(S)    # 状態空間の次元
        self.ndx    = len(DS)   # 微分行列の次元
        self.nu     = len(U)    # 制御入力の次元

        # 重み係数
        q = np.zeros(self.nx)
        s = np.zeros(self.nx)
        r = np.zeros(self.nu)
        
        q[int(S.ax)]    = 0.3
        q[int(S.ay)]    = 0.3
        q[int(S.xJerk)] = 0.4

        s[int(S.x)]     = 1.0
        s[int(S.y)]     = 1.0
        s[int(S.v)]     = 1.0
        
        self.Q = casadi.diag(q)
        self.S = casadi.diag(s)
        self.R = casadi.diag(r)

        path    = pd.read_csv(refFile)
        x       = path['x'].to_numpy()
        y       = path['y'].to_numpy()
        distance= path['Distance'].to_numpy()
        cur     = path['Curvature'].to_numpy()
        
        self.cur    = casadi.interpolant('interp', 'linear', [distance], cur)
        self.refX   = casadi.interpolant('interp', 'linear', [distance], x)
        self.refY   = casadi.interpolant('interp', 'linear', [distance], y)

        self.vehicle = Vehicle(self.cur)

        # 制約
        self.vmax       = 16.7
        self.vmin       = 0
        self.amax       = 2
        self.amin       = -1
        self.betamax    = 10*pi/180.0
        self.betamin    = -10*pi/180.0
        self.deltamax   = 40*pi/180.0
        self.deltamin   = -40*pi/180.0
        self.distmax    = 0.1
        self.distmin    = -0.1
        self.xJerkmax   = 1
        self.xJerkmin   = -1

        self.x_ub = [float('inf')] * self.nx
        
        self.x_ub[int(S.v)]     = self.vmax
        self.x_ub[int(S.a)]     = self.amax
        self.x_ub[int(S.beta)]  = self.betamax
        self.x_ub[int(S.delta)] = self.deltamax
        self.x_ub[int(S.dist)]  = self.distmax
        self.x_ub[int(S.xJerk)] = self.xJerkmax
        
        self.x_lb = [-float('inf')] * self.nx
        
        self.x_lb[int(S.v)]     = self.vmin
        self.x_lb[int(S.a)]     = self.amin
        self.x_lb[int(S.beta)]  = self.betamin
        self.x_lb[int(S.delta)]  = self.deltamin
        self.x_lb[int(S.dist)]  = self.distmin
        self.x_lb[int(S.xJerk)] = self.xJerkmin


        self.jerkmax    = 1
        self.jerkmin    = -1
        self.deltamax   = 12*pi/180.0
        self.deltamin   = -12*pi/180.0  
        
        self.u_ub = [self.jerkmax, self.deltamax]
        self.u_lb = [self.jerkmin, self.deltamin]

    def set_ref(self, ref):
        self.ref = ref

    def set_dt(self, dt):
        self.dt = dt
    
    def set_horizon(self, horizon):
        self.N = horizon

    def make_F(self):
        state   = casadi.MX.sym('state', self.nx)
        control = casadi.MX.sym('control', self.nu)
        
        state_next = self.vehicle.update_state(state, control, self.dt, self.ratio)
        
        F = casadi.Function("F", [state, control],[state_next],["x","u"],["x_next"])
        return F

    def stage_cost(self, x, u):
        return casadi.dot(self.Q@x,x)
    
    def terminal_cost(self, x):
        diff = x - self.ref
        cost = casadi.dot(self.S@diff,diff)
        return cost
    
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

        option  = {"print_time":False,"ipopt":{"print_level":0}}
        nlp     = {"x":casadi.vertcat(*X,*U),"f":J,"g":casadi.vertcat(*G)}
        self.S  = casadi.nlpsol("S","ipopt",nlp,option)
        return

    def compute_optimal_control(self,x_init,x0):
        x_init = x_init.full().ravel().tolist()

        lbx = x_init + self.x_lb*self.N + self.u_lb*self.N 
        ubx = x_init + self.x_ub*self.N + self.u_ub*self.N
        lbg = [0]*self.nx*self.N 
        ubg = [0]*self.nx*self.N 

        res     = self.S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0)
        offset  = self.nx*(self.N+1)
        
        x0      = res["x"]
        u_opt   = x0[offset:offset+self.nu]
        return u_opt, x0

