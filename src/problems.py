import numpy as np

def example_A():
    def f(t, y):
        return -2.0 * y
    t0 = 0.0
    T = 5.0
    y0 = 1.0
    exact = lambda t: np.exp(-2*t)
    return f, t0, T, y0, exact

def example_B():
    r = 3.0
    K = 2.0
    y0 = 0.2

    def f(t, y):
        return r*y*(1 - y/K)
    t0 = 0.0
    T = 5.0

    exact = lambda t: K / (1 + (K/y0 - 1)*np.exp(-r*t))
    return f, t0, T, y0, exact
