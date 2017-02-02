import numpy as np

from skopt import gp_minimize
from utils import black_box

# Provide initial points from 5.0 - 6.0, away from the original minimum.
x0 = np.reshape(np.linspace(5.0, 6.0, 10), (-1, 1)).tolist()
y0 = [black_box(xi) for xi in x0]

# Search for minimum from 0.0 - 6.0
dimensions = [[0.0, 6.0],]

# Optimize!
res = gp_minimize(
    black_box,
    dimensions=dimensions,
    x0=x0,
    y0=y0,
    n_random_starts=0,
    n_calls=4,
    random_state=0
)

print(res["x"])
[2.441]
