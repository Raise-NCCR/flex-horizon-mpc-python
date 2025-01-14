import numpy as np

from vehicleEnum import S

def rideComfortX(xs):
    xsTime = list(row[int(S.t)] for row in xs)

    x = [0]
    for i in range(len(xsTime)):
        if i == 0:
            continue
        
        b1 = 0.19
        b2 = 0.53
        b3 = 0.27
        b4 = 0.34

        cur_time = xsTime[i]
        trg = np.array(list(row for row in xs if cur_time - 3 <= row[int(S.t)] and row[int(S.t)] <= cur_time))
        ps = cur_time - trg[0][int(S.t)]

        trgA     = list(row[int(S.a)] for row in trg)
        ap = max(max(trgA),0)
        an = max(abs(min(trgA)),0)

        trgXjerk = np.arange(len(trg))
        for j in range(len(trg)):
            trgXjerk[j] = trg[j][int(S.xJerk)] * (trg[j][int(S.t)] - trg[j-1][int(S.t)])
        mean = np.sum(trgXjerk)/ps
        jr = np.sum(trgXjerk**2)/ps

        if mean >0:
            jr = b3*jr
        else:
            jr = b4*jr

        
        x.append(float(b1*ap+b2*an+jr[0]))
    
    return x
        