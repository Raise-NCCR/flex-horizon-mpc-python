import casadi
import numpy as np

from vehicleEnum import S, DS, U

class Vehicle:
    def __init__(self, refX, refY, cur):
        Mass        = 1100.0    # 車両重量
        YawMoment   = 1600.0    # ヨー慣性モーメント
        Cp_f        = 32000.0   # 前輪コーナリングパワー
        Cp_r        = 32000.0   # 後輪コーナリングパワー
        self.car_lf = 1.25      # 重心から前輪軸までの距離
        car_lr      = 1.25      # 重心から後輪軸までの距離


        self.a1 = -(Cp_f+Cp_r)/Mass
        self.a2 = (Cp_r*car_lr-Cp_f*self.car_lf)/Mass
        self.a3 = Cp_f/Mass
        self.a4 = -1/Mass
        self.b1 = (Cp_r*car_lr-Cp_f*self.car_lf)/YawMoment
        self.b2 = -(Cp_f*(self.car_lf**2)+Cp_r*((car_lr ** 2)))/YawMoment
        self.b3 = (Cp_f*self.car_lf)/YawMoment

        self.refX= refX    
        self.refY= refY
        self.cur = cur  # 曲率データ

    def kinematics(self, state, control, ds):
        d       = state[int(S.d)]
        v       = state[int(S.v)]
        a       = state[int(S.a)]
        beta    = state[int(S.beta)]
        delta   = state[int(S.delta)]
        omega   = state[int(S.omega)]
        psi     = state[int(S.psi)]
        theta   = state[int(S.theta)]
        dist    = state[int(S.dist)]
        omegaDot= state[int(S.omegaDot)]
        betaDot = state[int(S.betaDot)]

        jerk    = control[int(U.jerk)]

        cur     = self.cur(d)
        rho     = casadi.if_else(cur==0, np.inf, 1/cur)
        dDotTmp = v*casadi.cos(theta) - v*beta*casadi.sin(theta)

        sDot    = casadi.if_else(rho==np.inf, dDotTmp, rho*dDotTmp/(rho-dist))
        vDot    = a
        aDot    = jerk
        betaDot = casadi.atan(casadi.tan(delta)/2) - beta
        deltaDot   = control[int(U.deltaDot)]
        omegaDot= casadi.sin(beta)*v/self.car_lf - omega
        psiDot  = omega
        thetaDot= omega - sDot*cur
        xDot    = v*casadi.cos(psi+beta)
        yDot    = v*casadi.sin(psi+beta)
        distDot = v*casadi.sin(theta) + v*beta*casadi.cos(theta)
        ax      = a
        ay      = a*casadi.sin(beta)
        xJerk   = jerk
        yJerk   = jerk*casadi.sin(beta)
        dt      = sDot * ds / v
        return casadi.vertcat(vDot, aDot, betaDot, deltaDot, omegaDot, psiDot, thetaDot, xDot, yDot, distDot, ax, ay,xJerk, yJerk, dt) / sDot

    def dynamics(self, state, control, ds):
        d       = state[int(S.d)]
        v       = state[int(S.v)]
        a       = state[int(S.a)]
        beta    = state[int(S.beta)]
        delta   = state[int(S.delta)]
        omega   = state[int(S.omega)]
        psi     = state[int(S.psi)]
        theta   = state[int(S.theta)]
        dist    = state[int(S.dist)]
        
        jerk    = control[int(U.jerk)]

        cur     = self.cur(d)
        rho     = casadi.if_else(cur==0, np.inf, 1/cur)
        sDotTmp = v*casadi.cos(theta) - v*beta*casadi.sin(theta)        
        sDot    = casadi.if_else(rho==np.inf, sDotTmp, rho*sDotTmp/(rho+dist))
        
        vDot    = a
        aDot    = jerk
        betaDot = self.a1*beta/v + self.a2*omega/((v ** 2)) - omega + self.a3*delta/v + self.a4*a*beta/v
        deltaDot= control[int(U.deltaDot)]
        omegaDot= self.b1*beta + self.b2*omega/v + self.b3*delta
        psiDot  = omega
        thetaDot= omega - sDot*cur
        xDot    = (casadi.cos(psi) * v - casadi.sin(psi) * v * beta)
        yDot    = (casadi.sin(psi) * v + casadi.cos(psi) * v * beta)
        distDot = v*casadi.sin(theta) + v*beta*casadi.cos(theta)
        ax      = a - v*beta*omega
        ay      = v*betaDot + a*beta + v*omega
        xJerk   = jerk - a*beta*omega - v*betaDot*omega - v*beta*omegaDot
        yJerk   = a*betaDot + jerk*beta + a*betaDot + a*omega + v*omegaDot
        ax      = ax*sDot
        ay      = ay*sDot
        xJerk   = xJerk*sDot
        yJerk   = yJerk*sDot
        dt      = sDot * ds/v
        return casadi.vertcat(vDot, aDot, betaDot, deltaDot, omegaDot, psiDot, thetaDot, xDot, yDot, distDot, ax, ay,xJerk, yJerk, dt) / sDot
    

    def update(self, state, dstate, dt):
        state_next = state
        state_next  = [
            state_next[int(S.d)]    + dt,
            state_next[int(S.v)]    + dstate[int(DS.vDot)]*dt, 
            state_next[int(S.a)]    + dstate[int(DS.aDot)]*dt,
            state_next[int(S.beta)] + dstate[int(DS.betaDot)]*dt,
            state_next[int(S.delta)]+ dstate[int(DS.deltaDot)]*dt,
            state_next[int(S.omega)]+ dstate[int(DS.omegaDot)]*dt,
            state_next[int(S.psi)]  + dstate[int(DS.psiDot)]*dt,
            state_next[int(S.theta)]+ dstate[int(DS.thetaDot)]*dt,
            state_next[int(S.x)]    + dstate[int(DS.xDot)]*dt,
            state_next[int(S.y)]    + dstate[int(DS.yDot)]*dt,
            state_next[int(S.dist)] + dstate[int(DS.distDot)]*dt,
            dstate[int(DS.betaDot)],
            dstate[int(DS.omegaDot)],
            dstate[int(DS.ax)],
            dstate[int(DS.ay)],
            dstate[int(DS.xJerk)],
            dstate[int(DS.yJerk)],
            state_next[int(S.t)]    + dstate[int(DS.dt)],
        ]
        d = state_next[int(S.d)]
        next_d = d+1
        refXdot = self.refX(next_d) - self.refX(d)
        refYdot = self.refY(next_d) - self.refY(d)
        diffX   = state_next[int(S.x)] - self.refX(d)
        diffY   = state_next[int(S.y)] - self.refY(d)
        is_pos = casadi.if_else(diffX*refYdot - diffY*refXdot < 0, -1, 1)
        # is_pos = casadi.MX(1)
        state_next[int(S.dist)] = is_pos * casadi.sqrt(casadi.mmax(casadi.vertcat(casadi.MX(1e-10),(state_next[int(S.x)]-self.refX(d))**2+(state_next[int(S.y)]-self.refY(d))**2)))
        return state_next

    # 状態更新関数
    def update_state(self, state, control, dt):
        state_next = state

        
        dstate      = casadi.if_else(state_next[int(S.v)]>5, self.dynamics(state_next, control, dt), self.kinematics(state_next, control, dt))
        # k1 = dstate
        # k2_state = self.update(state, k1, dt/2)
        # k2_dstate = casadi.if_else(state_next[int(S.v)]>5, 
        #                    self.dynamics(k2_state, control, dt),
        #                    self.kinematics(k2_state, control, dt))
        # k3_state = self.update(state, k2_dstate, dt/2)
        # k3_dstate = casadi.if_else(state_next[int(S.v)]>5,
        #                    self.dynamics(k3_state, control, dt),
        #                    self.kinematics(k3_state, control, dt))
        # k4_state = self.update(state, k3_dstate, dt)
        # k4_dstate = casadi.if_else(state[int(S.v)]>5,
        #                    self.dynamics(k4_state, control, dt),
        #                    self.kinematics(k4_state, control, dt))
        # dstate = (k1 + 2*k2_dstate + 2*k3_dstate + k4_dstate)/6
        state_next = self.update(state, dstate, dt)
        # d = state_next[int(S.d)]
        # state_next[int(S.dist)] = (state_next[int(S.x)]-self.refX(d))**2+(state_next[int(S.y)]-self.refY(d))**2
        return casadi.vertcat(*state_next)
