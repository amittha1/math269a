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

def test_linear_problem(lambda_val, t0, T, y0):
    def f(t, y):
        return lambda_val * y
    exact = lambda t: y0 * np.exp(lambda_val * (t - t0))
    return f, t0, T, y0, exact

def example_C():
    def f(t,y):
        return 4.0 * t**2 * np.cos(y)
    def df_dy(t,y):
        return -4.0 * t**2 * np.sin(y)
    t0 = 0.0
    T = 4.0
    y0 = 0.1
    return f, df_dy, t0, T, y0

def example_D():

    A = np.array([[-2.0, 1.0],[1.0, -2.0]])
    def f(t, y):
        return A @ y
    t0 = 0.0
    T = 10.0
    y0 = np.array([1.0, 1.0])
    return f, t0, T, y0, A

def example_E():
    omega = 2.0
    A = np.array([[0.0, 1.0],[-omega**2, 0.0]])
    def f(t, y):
        return A @ y
    t0 = 0.0
    T = 10.0
    y0 = np.array([1.0, 0.0])
    def exact(t):
        return np.array([np.cos(omega * t), -omega * np.sin(omega * t)])
    return f, t0, T, y0, A, exact
