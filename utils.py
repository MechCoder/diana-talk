from math import cos

def black_box(x):
    """
    Function taken from https://www.mathworks.com/help/gads/example-finding-global-or-multiple-local-minima.html
    """
    r = x[0]
    return 2 + cos(r) + 0.5 * cos(2*r - 0.5)
