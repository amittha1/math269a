import numpy as np

def forward_euler(f, t0, T, y0, h):
    N = int((T - t0)/h)
    ts = np.linspace(t0, T, N+1)
    y0_arr = np.asarray(y0, dtype=float)
    ys = np.zeros((N+1,) + y0_arr.shape)
    ys[0] = y0_arr
    for i in range(N):
        ys[i+1] = ys[i] + h * f(ts[i], ys[i])
    return ts, ys

def rk2(f, t0, T, y0, h):
    N = int((T - t0)/h)
    ts = np.linspace(t0, T, N+1)
    y0_arr = np.asarray(y0, dtype=float)
    ys = np.zeros((N+1,) + y0_arr.shape)
    ys[0] = y0_arr
    for i in range(N):
        k1 = f(ts[i], ys[i])
        k2 = f(ts[i] + h, ys[i] + h * k1)
        ys[i+1] = ys[i] + (h/2) * (k1 + k2)
    return ts, ys

def rk4(f, t0, T, y0, h):
    N = int((T - t0)/h)
    ts = np.linspace(t0, T, N+1)
    y0_arr = np.asarray(y0, dtype=float)
    ys = np.zeros((N+1,) + y0_arr.shape)
    ys[0] = y0_arr
    for i in range(N):
        k1 = f(ts[i], ys[i])
        k2 = f(ts[i] + h/2, ys[i] + (h/2) * k1)
        k3 = f(ts[i] + h/2, ys[i] + (h/2) * k2)
        k4 = f(ts[i] + h, ys[i] + h * k3)
        ys[i+1] = ys[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return ts, ys

def newton_scalar(G, dG, y_init, tol, maxiter=20):
    y = y_init
    for k in range(1, maxiter + 1):
        Gy = G(y)
        if abs(Gy) <= tol:
            return y, k
        dGy = dG(y)
        if dGy == 0.0:
            raise RuntimeError("Derivative = 0, fail")
        y = y - Gy / dGy
    return y, maxiter


def backward_euler(f, df_dy, t0, T, y0, h, newton_tol, newton_maxiter=20):
    N = int((T - t0) / h)
    ts = np.linspace(t0, T, N + 1)
    ys = np.zeros(N + 1)
    ys[0] = y0
    newton_iters = np.zeros(N, dtype=int)

    for n in range(N):
        t_np1 = ts[n + 1]
        y_prev = ys[n]
        def G(y):
            return y - y_prev - h * f(t_np1, y)
        def dG(y):
            return 1.0 - h * df_dy(t_np1, y)
        y_guess = y_prev
        y_new, k = newton_scalar(G, dG, y_guess, newton_tol, newton_maxiter)
        ys[n + 1] = y_new
        newton_iters[n] = k

    return ts, ys, newton_iters


def trapezoidal(f, df_dy, t0, T, y0, h, newton_tol, newton_maxit=20):
    N = int((T - t0) / h)
    ts = np.linspace(t0, T, N + 1)
    ys = np.zeros(N + 1)
    ys[0] = y0
    newton_iters = np.zeros(N, dtype=int)

    for n in range(N):
        t_n = ts[n]
        t_np1 = ts[n + 1]
        y_prev = ys[n]
        f_prev = f(t_n, y_prev)
        def G(y):
            return y - y_prev - 0.5 * h * (f_prev + f(t_np1, y))

        def dG(y):
            return 1.0 - 0.5 * h * df_dy(t_np1, y)
        y_guess = y_prev
        y_new, k = newton_scalar(G, dG, y_guess, newton_tol, newton_maxit)
        ys[n + 1] = y_new
        newton_iters[n] = k

    return ts, ys, newton_iters
