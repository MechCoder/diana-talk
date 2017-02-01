from math import cos

import matplotlib.pyplot as plt
import numpy as np

from skopt.acquisition import gaussian_lcb

def black_box(x):
    """
    Function taken from https://www.mathworks.com/help/gads/example-finding-global-or-multiple-local-minima.html
    """
    r = x[0]
    return 2 + cos(r) + 0.5 * cos(2*r - 0.5)

def plot_space(prev_x, prev_y, model, x_cand):
    all_x = np.reshape(np.linspace(0, 6, 100), (-1, 1))
    all_f = [black_box(xi) for xi in all_x]
    plt.plot(all_x, all_f)

    plt.plot(np.ravel(prev_x), prev_y, "ro", label="Prev points")
    lcb_vals = gaussian_lcb(all_x, model)
    plt.plot(all_x, lcb_vals, "black", label="LCB")

    y_cand = black_box(x_cand)
    plt.plot([x_cand], [y_cand], "go", markersize=10, label="Next cand")
    plt.legend(numpoints=1)
    return plt
