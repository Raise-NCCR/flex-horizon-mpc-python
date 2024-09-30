from math import pi
import numpy as np
import pandas as pd
import casadi

from periodEnum     import S, U

class PeriodMPC:
    def __init__(self, N, curDiff, dest):
        # 問題設定
        self.N      = N         # ホライゾン離散化グリッド数
        self.nx     = len(S)    # 状態空間の次元
        self.nu     = len(U)    # 制御入力の次元

        # 重み係数
        q = np.zeros(self.nx)
        s = np.zeros(self.nx)
        r = np.zeros(self.nu)
        
        # q[int(S.var)]   = 1

        s[int(S.d)]     = 1/1000

        r[int(U.dDot)]  = 1/1000
        
        self.Q = casadi.diag(q)
        self.S = casadi.diag(s)
        self.R = casadi.diag(r)

        end = np.zeros(self.nx)
        end[int(S.d)] = dest + 20
        self.end = end

        self.curDiff    = curDiff

        # 制約
        varmax      = 0.006
        dmax        = dest + 20

        self.x_ub   = [float('inf')] * self.nx
        
        self.x_ub[int(S.var)]   = varmax
        self.x_ub[int(S.d)]     = dmax
        
        self.x_lb = [-float('inf')] * self.nx

        dDotmin    = 1.0
        
        self.u_ub = [float('inf')] * self.nu
        self.u_lb = [-float('inf')] * self.nu

        self.u_lb[int(U.dDot)] = dDotmin

    def make_F(self):
        state   = casadi.MX.sym('state', self.nx)
        control = casadi.MX.sym('control', self.nu)
        
        state_next = self.update_state(state, control)
        
        F = casadi.Function("F", [state, control],[state_next],["x","u"],["x_next"])
        return F
    
    def update_state(self, state, control):
        n = 10
        new_d = state[int(S.d)] + control[int(U.dDot)]
        ds = casadi.linspace(state[int(S.d)], new_d, n)
        cur = self.curDiff(ds)
        mean_cur = casadi.cumsum(cur)/n
        diff = cur - mean_cur
        var = casadi.dot(diff,diff)/n
        state_next = [new_d, var]
        return casadi.vertcat(*state_next)

    def stage_cost(self, x, u):
        # cost = casadi.if_else(u[int(U.dDot)] == 0, 100, self.R/casadi.dot(u,u))
        cost = 1/casadi.dot(u,u)
        return cost
    
    def terminal_cost(self, x, x0):
        # diff = x - self.end
        diff = x - x0
        cost = 1/casadi.dot(diff,diff)
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
        J += self.terminal_cost(X[self.N], X[0])

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

