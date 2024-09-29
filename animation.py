import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, cos, sin

from myEnum import S, U

vehicle_length = 2.5
vehicle_width = 1.5
vehicle_d = np.sqrt(2.5**2+1.5**2)

# 車両の座標データ（例としてランダムなデータを生成）
df = pd.read_csv('result/mpcX.csv')
x_data = df['x'].to_numpy()
y_data = df['y'].to_numpy()
theta_data = df['psi'].to_numpy()
theta_data = list(map(lambda theta: theta*180.0/pi, theta_data))

df = pd.read_csv('csv/disCur.csv')
zhouX = df['x'].to_numpy()
zhouY = df['y'].to_numpy()

xx = np.load('result/xx.npy')

i_x = int(S.x)
i_y = int(S.y)
len_xx = len(S)+len(U)
end = len(S) * 16

# 描画領域の準備
fig, ax = plt.subplots()
ax.set_xlim(-5, 100)
ax.set_ylim(-5, 100)

plt.plot(zhouX, zhouY, '-')

# 車両を表す四角形を描く
vehicle = plt.Rectangle((x_data[0]-vehicle_length/2, y_data[0]-vehicle_width/2), vehicle_length, vehicle_width, fc='blue')
ax.add_patch(vehicle)
line, = ax.plot([],[], 'r-')

# アニメーションの更新関数
def update(frame):
    # 現在の座標に四角形を移動
    xx_X = xx[frame-1][len(S)+i_x:end:len(S)]
    xx_Y = xx[frame-1][len(S)+i_y:end:len(S)]
    theta = theta_data[frame]
    x_diff = vehicle_length/2*cos(theta*pi/180) - vehicle_width/2*sin(theta*pi/180)
    y_diff = vehicle_length/2*sin(theta*pi/180) + vehicle_width/2*cos(theta*pi/180)
    vehicle.set_xy([x_data[frame]-x_diff, y_data[frame]-y_diff])
    vehicle.set_angle(theta)
    line.set_data(xx_X, xx_Y)
    return vehicle, line

# アニメーションの作成
ani = FuncAnimation(fig, update, frames=len(x_data), interval=100, blit=True)

# 保存する場合 (例: "vehicle_movement.mp4"に保存)
ani.save('fig/movie/vehicle_movement.gif', writer='imagemagick')

# アニメーションを表示
plt.show()