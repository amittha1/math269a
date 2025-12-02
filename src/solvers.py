import numpy as np

def forward_euler(f, t0, T, y0, h):
    N = int((T - t0)/h)
    ts = np.linspace(t0, T, N+1)
    ys = np.zeros(N+1)
    ys[0] = y0
    for i in range(N):
        ys[i+1] = ys[i] + h * f(ts[i], ys[i])
    return ts, ys
