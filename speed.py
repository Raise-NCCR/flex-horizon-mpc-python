import casadi
import numpy as np

from speedEnum import S, DS, U

class Speed:
    def __init__(self, speed):
        self.tau    = 1
        self.speed = speed

    def dynamics(self, state, control):
        a       = state[int(S.a)]
        v       = state[int(S.v)]
        
        a_input    = control[int(U.a)]

        vDot    = a_input
        aDot    = (a_input - a)/self.tau
        return casadi.vertcat(vDot, aDot)/v
    

    def update(self, state, dstate, dt):
        state_next = state
        state_next  = [
            state_next[int(S.d)]    + dt,
            state_next[int(S.v)]    + dstate[int(DS.vDot)]*dt, 
            state_next[int(S.a)]    + dstate[int(DS.aDot)]*dt,
            self.speed(state_next[int(S.d)]) - state_next[int(S.v)],
        ]
        return state_next

    # 状態更新関数
    def update_state(self, state, control, dt):
        state_next = state
 
        dstate      = self.dynamics(state_next, control)
        state_next = self.update(state, dstate, dt)
        return casadi.vertcat(*state_next)
