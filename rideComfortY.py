import numpy as np

from vehicleEnum import S

def rideComfortY(xs):
    xsTime = list(row[int(S.t)] for row in xs)

    y = [0]
    for i in range(len(xsTime)):
        if i == 0:
            continue
        
        c1 = 1.08
        c2 = 0.125

        cur_time = xsTime[i]
        trg = list(row for row in xs if cur_time - 3 <= row[int(S.t)] and row[int(S.t)] <= cur_time)
        ps = cur_time - trg[0][int(S.t)]

        trgA = np.arange(len(trg))
        for j in range(len(trg)):
            trgA[j] = trg[j][int(S.ay)] * (trg[j][int(S.t)] - trg[j-1][int(S.t)])
        ar = np.sum(trgA**2)/ps

        trgJerk = np.arange(len(trg))
        for j in range(len(trg)):
            trgJerk[j] = trg[j][int(S.yJerk)] * (trg[j][int(S.t)] - trg[j-1][int(S.t)])
        jr = np.sum(trgJerk**2)/ps
        
        y.append(float(c1*ar[0]+c2*jr[0]))
    
    return y