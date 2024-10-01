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
        omega   = state[int(S.omega)]
        psi     = state[int(S.psi)]
        theta   = state[int(S.theta)]
        omegaDot= state[int(S.omegaDot)]
        betaDot = state[int(S.betaDot)]

        jerk    = control[int(U.jerk)]
        delta   = control[int(U.delta)]

        cur     = self.cur(d)
        sDot = v*casadi.cos(theta) - v*beta*casadi.sin(theta)

        vDot    = a
        aDot    = jerk
        betaDot = casadi.atan(casadi.tan(delta)/2) - beta
        omegaDot= casadi.sin(beta)*v/self.car_lf - omega
        psiDot  = omega
        thetaDot= omega - sDot*cur
        xDot    = v*casadi.cos(psi+beta)
        yDot    = v*casadi.sin(psi+beta)
        ax      = a
        ay      = a*casadi.sin(beta)
        xJerk   = jerk
        yJerk   = jerk*casadi.sin(beta)
        dt      = sDot * ds / v
        return casadi.vertcat(vDot, aDot, betaDot, omegaDot, psiDot, thetaDot, xDot, yDot, ax, ay,xJerk, yJerk, dt) / sDot

    def dynamics(self, state, control, ds):
        d       = state[int(S.d)]
        v       = state[int(S.v)]
        a       = state[int(S.a)]
        beta    = state[int(S.beta)]
        omega   = state[int(S.omega)]
        psi     = state[int(S.psi)]
        theta   = state[int(S.theta)]
        
        jerk    = control[int(U.jerk)]
        delta   = control[int(U.delta)]

        cur     = self.cur(d)
        sDot = v*casadi.cos(theta) - v*beta*casadi.sin(theta)
        
        vDot    = a
        aDot    = jerk
        betaDot = self.a1*beta/v + self.a2*omega/((v ** 2)) - omega + self.a3*delta/v + self.a4*a*beta/v
        omegaDot= self.b1*beta + self.b2*omega/v + self.b3*delta
        psiDot  = omega
        thetaDot= omega - sDot*cur
        xDot    = (casadi.cos(psi) * v - casadi.sin(psi) * v * beta)
        yDot    = (casadi.sin(psi) * v + casadi.cos(psi) * v * beta)
        ax      = a - v*beta*omega
        ay      = v*betaDot + a*beta + v*omega
        xJerk   = jerk - a*beta*omega - v*betaDot*omega - v*beta*omegaDot
        yJerk   = a*betaDot + jerk*beta + a*betaDot + a*omega + v*omegaDot
        ax      = ax*sDot
        ay      = ay*sDot
        xJerk   = xJerk*sDot
        yJerk   = yJerk*sDot
        dt      = sDot * ds/v
        return casadi.vertcat(vDot, aDot, betaDot, omegaDot, psiDot, thetaDot, xDot, yDot, ax, ay,xJerk, yJerk, dt) / sDot

    # 状態更新関数
    def update_state(self, state, control, dt):
        dd      = 1
        
        state_next = state

        ddt = dt/dd
        dstate      = casadi.if_else(state_next[int(S.v)]>5, self.dynamics(state_next, control, ddt), self.kinematics(state_next, control, ddt))
        state_next  = [
                        state_next[int(S.d)]    + ddt,
                        state_next[int(S.v)]    + dstate[int(DS.vDot)]*ddt, 
                        state_next[int(S.a)]    + dstate[int(DS.aDot)]*ddt,
                        state_next[int(S.beta)] + dstate[int(DS.betaDot)]*ddt,
                        state_next[int(S.omega)]+ dstate[int(DS.omegaDot)]*ddt,
                        state_next[int(S.psi)]  + dstate[int(DS.psiDot)]*ddt,
                        state_next[int(S.theta)]+ dstate[int(DS.thetaDot)]*ddt,
                        state_next[int(S.x)]    + dstate[int(DS.xDot)]*ddt,
                        state_next[int(S.y)]    + dstate[int(DS.yDot)]*ddt,
                        0,
                        dstate[int(DS.betaDot)],
                        dstate[int(DS.omegaDot)],
                        dstate[int(DS.ax)],
                        dstate[int(DS.ay)],
                        dstate[int(DS.xJerk)],
                        dstate[int(DS.yJerk)],
                        state_next[int(S.t)]    + dstate[int(DS.dt)],
                        dt,
                    ]
        d = state_next[int(S.d)]
        state_next[int(S.dist)] = (state[int(S.x)]-self.refX(d))**2+(state[int(S.y)]-self.refY(d))**2
        return casadi.vertcat(*state_next)
