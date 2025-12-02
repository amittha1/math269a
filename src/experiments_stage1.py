import numpy as np
import matplotlib.pyplot as plt

from .solvers import forward_euler
from .problems import example_A

def run_stage1():
    f, t0, T, y0, exact = example_A()

    hs = [2**(-k) for k in range(3, 10)]
    errors = []

    for h in hs:
        ts, ys = forward_euler(f, t0, T, y0, h)
        err = abs(ys[-1] - exact(T))
        errors.append(err)

    plt.loglog(hs, errors, 'o-')
    plt.xlabel("h")
    plt.ylabel("error")
    plt.title("Forward Euler Convergence (Example A)")
    plt.savefig("figs/stage1_forward_euler.png", dpi=150)
    plt.close()
