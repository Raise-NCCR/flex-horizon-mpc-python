import pandas as pd
import numpy as np

from myEnum import S, U

i_x = int(S.x)
i_y = int(S.y)
end = len(S) * 16

xx = np.load('result/xx.npy')
xx_X = xx[1][len(S)+i_x:end:len(S)]
xx_Y = xx[1][len(S)+i_y:end:len(S)]
print(xx_X)
# print(xx[1][int(S.x)], xx[1][int(S.y)])
