import numpy as np
import matplotlib.pyplot as plt

from utils import black_box

all_x = np.reshape(np.linspace(0, 6, 100), (-1, 1))
all_f = [black_box(xi) for xi in all_x]
plt.plot(all_x, all_f)
plt.show()
