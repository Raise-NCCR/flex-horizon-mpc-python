import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_path():
    # 1. 100メートル直進
    x1 = np.arange(0, 99.9, 0.1)
    y1 = np.zeros_like(x1)

    # 左折
    theta = np.linspace(-np.pi/2, 0, 101)  # 60度分の点を生成
    x2 = 30 * (-np.cos(-np.pi/2) + np.cos(theta)) + x1[-1]# x座標を100だけずらして接続
    y2 = 30 * (-np.sin(-np.pi/2) + np.sin(theta)) + y1[-1]

    # 2. 直進
    y3 = np.arange(y2[-1], y2[-1]+100.1, 0.1)
    x3 = np.full_like(y3, x2[-1])
    print(len(x3), len(y3))

    # 左折
    theta = np.linspace(0, np.pi/2, 101)  # 60度分の点を生成
    x4 = 30 * (-np.cos(0) + np.cos(theta)) + x3[-1] # x座標を100だけずらして接続
    y4 = 30 * np.sin(theta) + y3[-1]

    # 右折
    theta = np.linspace(0, -np.pi/2, 101)  # 60度分の点を生成
    x4 = 30 * (-np.cos(0) + np.cos(theta)) + x3[-1] # x座標を100だけずらして接続
    y4 = 30 * np.sin(theta) + y3[-1]

    # 3. 100メートル直進
    x5 = np.arange(x4[-1], x4[-1]-100.1, -0.1)
    y5 = np.full_like(x5, y4[-1])

    # # 4. 右折
    # x4 = np.zeros(1001)
    # y4 = np.arange(100, 200.1, 0.1)

    # # 5. 100メートル直進
    # x5 = np.arange(0, 100.1, 0.1)
    # y5 = np.full_like(x5, 200)

    # 全ての点を結合
    # x = np.concatenate([x1, x2, x3, x4, x5])
    # y = np.concatenate([y1, y2, y3, y4, y5])
    x = np.concatenate([x1[:-1], x2[:-1], x3[:-1], x4[:-1], x5])
    y = np.concatenate([y1[:-1], y2[:-1], y3[:-1], y4[:-1], y5])

    return x, y

# 経路を生成
x, y = generate_path()

dx = np.gradient(x)
dy = np.gradient(y)

# 2次微分 (加速度ベクトル)
ddx = np.gradient(dx)
ddy = np.gradient(dy)

# 曲率の計算
curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5


# 距離を計算
distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
total_distances = np.cumsum(distances)
total_distances = np.insert(total_distances, 0, 0)  # 距離の最初の点を0に設定

# 距離と曲率データを新しいDataFrameにまとめる
output_df = pd.DataFrame({
    'Distance': total_distances,
    'x': x,
    'y': y,
    'Curvature': curvature
})

# 結果を新しいCSVファイルに保存
output_df.to_csv('csv/genPath.csv', index=False)

# グラフで経路を可視化
# plt.figure(figsize=(10, 10))
# plt.plot(x, y, 'b-')
# plt.title('自動車の運動シミュレーション経路')
# plt.xlabel('X座標 (m)')
# plt.ylabel('Y座標 (m)')
# plt.grid(True)
# plt.axis('equal')
# plt.show()