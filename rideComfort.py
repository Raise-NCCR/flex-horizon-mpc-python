from rideComfortX import rideComfortX
from rideComfortY import rideComfortY

def rideComfort(xs):
    x = rideComfortX(xs)
    y = rideComfortY(xs)
    comfort = []
    for i in range(len(x)):
        comfort.append(x[i]+y[i])
    return comfort

