import matplotlib.pyplot as plt
import numpy as np

plt.figure()

# 按2行1列分，这个占一行
plt.subplot(2, 1, 1)
plt.plot([0, 1], [0, 1])

# 按2行3列分，这个在第4个位置，后面依次
plt.subplot(212)
plt.plot([0, 1], [0, 2])

plt.show()
