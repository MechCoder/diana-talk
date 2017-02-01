from utils import black_box
from utils import plot_space

import numpy as np
from skopt.optimizer import Optimizer
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from skopt.benchmarks import branin

# Search from 0.0 to 6.0
dimensions = ((0.0, 6.0),)

# Initialize estimator.
gpr = GaussianProcessRegressor(kernel=RBF(), noise=0.0)
optimizer = Optimizer(
    dimensions=dimensions,
    base_estimator=gpr,
    n_random_starts=0,
    acq_func="LCB",
    random_state=0)

# Tell some points to the optimizer.
X = np.reshape(np.linspace(4, 6, 10), (-1, 1)).tolist()
y = [black_box(xi) for xi in X]
optimizer.tell(X, y)
x_cand = optimizer.ask()
y_cand = black_box(x_cand)

plot = plot_space(X, y, optimizer.models[-1], x_cand)
plot.show()
