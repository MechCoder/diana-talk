from math import sin

def black_box(x):
    """
    Function taken from https://www.mathworks.com/help/gads/example-finding-global-or-multiple-local-minima.html
    """
    r = x[0]
    return r**2 * ((sin(r) - (sin(2*r) / 2) + (sin(3*r) / 3) - (sin(4*r) / 4) + 4)) / (r + 1)
