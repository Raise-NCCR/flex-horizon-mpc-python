from enum import IntEnum, auto

class S(IntEnum):
    d       = 0
    v       = auto()
    a       = auto()
    vErr    = auto()
    
class DS(IntEnum):
    vDot    = 0
    aDot    = auto()
    
class U(IntEnum):
    a       = 0
