import numpy as np
import matplotlib.pyplot as plt

from utils import black_box

x = np.reshape(np.linspace(0, 20, 100), (-1, 1))
f = [black_box(xi) for xi in x]
plt.plot(x, f)
plt.show()
