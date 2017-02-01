from utils import black_box

import matplotlib.pyplot as plt
import numpy as np
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from skopt.acquisition import gaussian_ei
from skopt.acquisition import gaussian_lcb

all_x = np.reshape(np.linspace(0, 6, 100), (-1, 1))
all_f = [black_box(xi) for xi in all_x]

# Plot all points.
plt.plot(all_x, all_f, "green", label="Ground truth")

# Train only one third of the training data.
X = np.reshape(np.linspace(4, 6, 10), (-1, 1))
y = [black_box(xi) for xi in X]

# Use RBF kernel.
rbf = RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, alpha=1e-12)
gpr.fit(X, y)
y_pred, y_std = gpr.predict(all_x, return_std=True)

ei_vals = -gaussian_ei(all_x, gpr, y_opt=np.min(y))
lcb_vals = gaussian_lcb(all_x, gpr)
all_x_plot = np.ravel(all_x)
upper_bound = y_pred + 1.96*y_std
lower_bound = y_pred - 1.96*y_std

plt.title("Acquisition values.")
plt.plot(all_x_plot, y_pred, "r", label="Predictions")
plt.plot(all_x_plot, ei_vals, "b", label="-EI")
plt.plot(all_x_plot, lcb_vals, "black", label="LCB")
plt.legend()
plt.show()
