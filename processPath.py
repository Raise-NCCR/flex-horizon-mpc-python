import pandas as pd
import numpy as np

# CSVファイルを読み込み（headerなし）
df = pd.read_csv('zhouPath.csv', header=None)

# 1,2行目にあるx, y座標のデータを抽出
x = df.iloc[0].values
y = df.iloc[1].values

# 曲率データ（3行目）
curvature = df.iloc[2].values
#cur = np.zeros(len(curvature))
#for i in range(len(curvature)):
#    if curvature[i] == 0:
#        cur[i] = 0
#    else:
#        cur[i] = 1/curvature[i]


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
output_df.to_csv('newPathC.csv', index=False)

