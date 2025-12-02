import numpy as np

def example_A():
    def f(t, y):
        return -2.0 * y
    t0 = 0.0
    T = 5.0
    y0 = 1.0
    exact = lambda t: np.exp(-2*t)
    return f, t0, T, y0, exact
