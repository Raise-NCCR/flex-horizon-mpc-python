from math import pi
import numpy as np
import pandas as pd
import casadi

from periodEnum     import S, U

class PeriodMPC:
    def __init__(self, N, cur, acc, dest):
        # 問題設定
        self.N      = N         # ホライゾン離散化グリッド数
        self.nx     = len(S)    # 状態空間の次元
        self.nu     = len(U)    # 制御入力の次元

        # 重み係数
        q = np.zeros(self.nx)
        s = np.zeros(self.nx)
        r = np.zeros(self.nu)
        
        q[int(S.var)]   = 1000000
        # q[int(S.var_acc)]   = 1

        # s[int(S.d)]     = 1000

        # r[int(U.dDot)]  = 1000
        
        self.Q = casadi.diag(q)
        self.S = casadi.diag(s)
        self.R = casadi.diag(r)

        end = np.zeros(self.nx)
        end[int(S.d)] = dest + 20
        self.end = end

        self.acc    = acc
        self.cur    = cur
        
        # 制約
        dmax        = dest + 20

        self.x_ub   = [float('inf')] * self.nx
        
        self.x_ub[int(S.d)]     = dmax
        
        self.x_lb = [-float('inf')] * self.nx

        dDotMax     = 10.0
        dDotmin     = 1.0
        
        self.u_ub = [float('inf')] * self.nu
        self.u_lb = [-float('inf')] * self.nu

        self.u_ub[int(U.dDot)] = dDotMax
        self.u_lb[int(U.dDot)] = dDotmin

    def make_F(self):
        state   = casadi.MX.sym('state', self.nx)
        control = casadi.MX.sym('control', self.nu)
        
        state_next = self.update_state(state, control)
        
        F = casadi.Function("F", [state, control],[state_next],["x","u"],["x_next"])
        return F
    
    def update_state(self, state, control):
        n = 10
        # dDot = state[int(S.dDot)] + (control[int(U.dDot)]-state[int(S.dDot)])/3
        dDot = control[int(U.dDot)]
        new_d = state[int(S.d)] + dDot
        ds = casadi.linspace(state[int(S.d)], new_d, n)
        cur = self.cur(ds)
        acc = self.acc(ds)
        var_cur = (casadi.sum1(cur) + casadi.sum1(acc)) * control[int(U.dDot)] # 誤差に関するペナルティをdt^2と曲率の2乗和の積で表現
        
        # mean_cur= casadi.cumsum(cur)/n
        # diff_cur = cur - mean_cur
        # var_cur = dDot*casadi.dot(diff_cur,diff_cur)/n
        # mean_acc = casadi.cumsum(acc)/n
        # diff_acc = acc - mean_acc
        # var_acc = dDot*casadi.dot(diff_acc,diff_acc)/n
        state_next = [new_d, dDot, var_cur]
        return casadi.vertcat(*state_next)

    def stage_cost(self, x, u):
        # cost = casadi.if_else(u[int(U.dDot)] == 0, 100, self.R/casadi.dot(u,u))
        # cost = casadi.sum1(self.Q@x)
        cost = 1000000*x[int(S.var)]**2 + (3.0 - x[int(S.dDot)])**2
        return cost
    
    def terminal_cost(self, x, x0):
        # diff = x - self.end
        diff = x - x0
        # cost = casadi.dot(diff[int(S.d)],diff[int(S.d)])
        cost = 0
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

        option  = {"print_time":False,"ipopt":{"print_level":0},"max_iter_eig":10}
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

