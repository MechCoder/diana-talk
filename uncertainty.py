from utils import black_box

import matplotlib.pyplot as plt
import numpy as np
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF

all_x = np.reshape(np.linspace(0, 20, 100), (-1, 1))
all_f = [black_box(xi) for xi in all_x]

# Plot all points.
plt.plot(all_x, all_f)

# Train only one half of the training data.
X = np.reshape(np.linspace(0, 10, 10), (-1, 1))
y = [black_box(xi) for xi in X]

# Use RBF kernel.
rbf = RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-12)
gpr.fit(X, y)
plt.plot(np.ravel(X), y, "ro", label="True function")

# Predict on all data.
y_pred, y_std = gpr.predict(all_x, return_std=True)
all_x_plot = np.ravel(all_x)
upper_bound = y_pred + 1.96*y_std
lower_bound = y_pred - 1.96*y_std

plt.plot(all_x_plot, y_pred, "r--", label="Predictions")
plt.plot(all_x_plot, lower_bound, color="red")
plt.plot(all_x_plot, upper_bound, color="red")
plt.fill_between(all_x_plot, lower_bound, upper_bound, facecolor="lightcoral")
plt.legend()
plt.show()
