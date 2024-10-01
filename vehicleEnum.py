from enum import IntEnum, auto

class S(IntEnum):
    d       = 0
    v       = auto()
    a       = auto()
    beta    = auto()
    omega   = auto()
    psi     = auto()
    theta   = auto()
    x       = auto()
    y       = auto()
    dist    = auto()
    ax      = auto()
    ay      = auto()
    betaDot = auto()
    omegaDot= auto()
    xJerk   = auto()
    yJerk   = auto()
    t       = auto()
    dt      = auto()

class DS(IntEnum):
    vDot    = 0
    aDot    = auto()
    betaDot = auto()
    omegaDot= auto()
    psiDot  = auto()
    thetaDot= auto()
    xDot    = auto()
    yDot    = auto()
    ax      = auto()
    ay      = auto()
    xJerk   = auto()
    yJerk   = auto()
    dt      = auto()

class U(IntEnum):
    jerk    = 0
    delta   = auto()
