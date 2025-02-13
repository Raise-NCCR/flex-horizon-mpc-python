from math import pi
import numpy as np
import pandas as pd
import casadi
from scipy import interpolate
from matplotlib import pyplot as plt

from speedEnum     import S, DS, U
from speed    import Speed

class SpeedMPC:
    def __init__(self, refFile, N):
        # 問題設定
        self.N      = N         # ホライゾン離散化グリッド数 (MPCなので荒め)
        self.nx     = len(S)    # 状態空間の次元
        self.ndx    = len(DS)   # 微分行列の次元
        self.nu     = len(U)    # 制御入力の次元

        # 重み係数
        q = np.zeros(self.nx)
        s = np.zeros(self.nx)
        r = np.ones(self.nu)
        
        q[int(S.vErr)] = 1

        self.Q = casadi.diag(q)
        self.S = casadi.diag(s)
        self.R = casadi.diag(r)

        path    = pd.read_csv(refFile)
        distance= path['Distance'].to_numpy()
        speed   = path['Speed'].to_numpy()
        
        self.speed  = casadi.interpolant('interp', 'linear', [distance], speed)

        self.model = Speed(self.speed)

        # 制約
        self.vmax       = 60/3.6
        self.vmin       = 0
        self.vErrmin    = 0

        self.aInputmax  = 2
        self.aInputmin  = -2

        self.x_ub = [float('inf')] * self.nx
        
        self.x_ub[int(S.v)]     = self.vmax
        
        self.x_lb = [-float('inf')] * self.nx
        
        self.x_lb[int(S.v)]     = self.vmin
        self.x_lb[int(S.vErr)] = self.vErrmin
        
        self.u_ub = [float('inf')] * self.nu
        self.u_lb = [-float('inf')] * self.nu

        self.u_ub[int(U.a)] = self.aInputmax
        self.u_lb[int(U.a)] = self.aInputmin


    def make_F(self):
        state   = casadi.MX.sym('state', self.nx)
        control = casadi.MX.sym('control', self.nu)
        dt      = casadi.MX.sym('dt', 1)
        
        state_next = self.model.update_state(state, control, dt)
        
        F = casadi.Function("F", [state, control, dt],[state_next],["x","u","p"],["x_next"])
        return F

    def stage_cost(self, x, u):
        return casadi.dot(self.Q@x,x)+casadi.dot(self.R@u, u)
    
    def terminal_cost(self, x):
        cost = casadi.dot(self.S@x,x)
        return cost
    
    def make_nlp(self):
        F = self.make_F()

        X = [casadi.MX.sym(f"x_{k}",self.nx) for k in range(self.N+1)]
        U = [casadi.MX.sym(f"u_{k}",self.nu) for k in range(self.N)]
        G = []
        P = [casadi.MX.sym(f"p_{k}",1) for k in range(self.N)]

        J = 0
        for k in range(self.N):
            J += self.stage_cost(X[k],U[k])
            eq = X[k+1] - F(x=X[k],u=U[k],p=P[k])["x_next"]
            print(eq)
            G.append(eq)
        J += self.terminal_cost(X[self.N])

        option  = {"print_time":False,"ipopt":{"print_level":0}}
        nlp     = {"x":casadi.vertcat(*X,*U),"f":J,"g":casadi.vertcat(*G),"p":casadi.vertcat(*P)}
        self.S  = casadi.nlpsol("S","ipopt",nlp,option)
        return

    def compute_optimal_control(self,x_init,x0,dt):
        x_init = x_init.full().ravel().tolist()

        lbx = x_init + self.x_lb*self.N + self.u_lb*self.N 
        ubx = x_init + self.x_ub*self.N + self.u_ub*self.N
        lbg = ([0.0]*self.nx)*self.N 
        ubg = ([0.0]*self.nx)*self.N

        p = dt
        
        res     = self.S(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, x0=x0, p=p)
         
        x       = res["x"]
        u       = x[self.nx*(self.N+1)]
        v   = x[int(S.v):int(S.v)+self.nx*self.N:self.nx]
        return u, v, x