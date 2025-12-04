import numpy as np

def forward_euler(f, t0, T, y0, h):
    N = int((T - t0)/h)
    ts = np.linspace(t0, T, N+1)
    ys = np.zeros(N+1)
    ys[0] = y0
    for i in range(N):
        ys[i+1] = ys[i] + h * f(ts[i], ys[i])
    return ts, ys

def rk2(f, t0, T, y0, h):
    N = int((T - t0)/h)
    ts = np.linspace(t0, T, N+1)
    ys = np.zeros(N+1)
    ys[0] = y0
    for i in range(N):
        k1 = f(ts[i], ys[i])
        k2 = f(ts[i] + h, ys[i] + h * k1)
        ys[i+1] = ys[i] + (h/2) * (k1 + k2)
    return ts, ys

def rk4(f, t0, T, y0, h):
    N = int((T - t0)/h)
    ts = np.linspace(t0, T, N+1)
    ys = np.zeros(N+1)
    ys[0] = y0
    for i in range(N):
        k1 = f(ts[i], ys[i])
        k2 = f(ts[i] + h/2, ys[i] + (h/2) * k1)
        k3 = f(ts[i] + h/2, ys[i] + (h/2) * k2)
        k4 = f(ts[i] + h, ys[i] + h * k3)
        ys[i+1] = ys[i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return ts, ys