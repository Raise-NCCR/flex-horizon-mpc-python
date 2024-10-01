import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, cos, sin
from scipy import interpolate

from vehicleEnum import S, U

dt = 1
vehicle_length = 5
vehicle_width = 3
vehicle_d = np.sqrt(5**2+3**2)

# 車両の座標データ（例としてランダムなデータを生成）
df = pd.read_csv('result/mpcX.csv')
x_data = df['x'].to_numpy()
y_data = df['y'].to_numpy()
t_data = df['t'].to_numpy()
theta_data = df['psi'].to_numpy()
theta_data = list(map(lambda theta: theta*180.0/pi, theta_data))

end_t = t_data[-1]
length = int(end_t/dt+1)

x_data = interpolate.interp1d(t_data,x_data)
y_data = interpolate.interp1d(t_data,y_data)
theta_data = interpolate.interp1d(t_data,theta_data)

df = pd.read_csv('csv/disCur_path.csv')
zhouX = df['x'].to_numpy()
zhouY = df['y'].to_numpy()

# 描画領域の準備
fig, ax = plt.subplots()
ax.set_xlim(-5, max(zhouX)+10)
ax.set_ylim(-5, max(zhouY)+10)

plt.plot(zhouX, zhouY, '-')

# 車両を表す四角形を描く
vehicle = plt.Rectangle((x_data(0)-vehicle_length/2, y_data(0)-vehicle_width/2), vehicle_length, vehicle_width, fc='red')
ax.add_patch(vehicle)

# アニメーションの更新関数
def update(frame):
    # 現在の座標に四角形を移動
    theta = theta_data(frame*dt)
    x_diff = vehicle_length/2*cos(theta*pi/180) - vehicle_width/2*sin(theta*pi/180)
    y_diff = vehicle_length/2*sin(theta*pi/180) + vehicle_width/2*cos(theta*pi/180)
    vehicle.set_xy([x_data(frame*dt)-x_diff, y_data(frame*dt)-y_diff])
    vehicle.set_angle(theta)
    return vehicle,


# アニメーションの作成
ani = FuncAnimation(fig, update, frames=length, interval=100, blit=True)

# 保存する場合 (例: "vehicle_movement.mp4"に保存)
ani.save('fig/movie/vehicle_movement_t.gif', writer='imagemagick')

# アニメーションを表示
plt.show()