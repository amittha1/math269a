import numpy as np
import matplotlib.pyplot as plt

from .problems import example_A 
from .solvers import rk4  


def rk4_step(f, t, y, h):
    y = np.asarray(y, dtype=float)
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h,       y + h * k3)
    return y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def ab2_step(f, t_n, y_n, y_nm1, h):
    f_n   = f(t_n,     y_n)
    f_nm1 = f(t_n - h, y_nm1)
    return y_n + 0.5*h*(3.0*f_n - f_nm1)

def am2_step(f, t_n, y_n, y_nm1, h, newton_it=5):
    f_n   = f(t_n,     y_n)
    f_nm1 = f(t_n - h, y_nm1)
    y_pred = y_n + 0.5*h*(3.0*f_n - f_nm1)
    t_np1 = t_n + h
    y = y_pred
    for _ in range(newton_it):
        f_np1 = f(t_np1, y)
        eps = 1e-8
        fy_approx = (f(t_np1, y + eps) - f_np1) / eps
        G  = y - y_n - 0.5*h*(f_np1 + f_n)
        dG = 1.0 - 0.5*h*fy_approx
        y -= G / dG
    return y

def stage8_zero_stability():
    def f_zero(t, y):
        return 0.0 * y  
    
    h = 0.1
    N = 500           
    t = np.linspace(0.0, N*h, N+1)
    y_ab = np.zeros(N+1)
    y_ab[0] = 1.0e-4
    y_ab[1] = 2.0e-4   
    for n in range(1, N):
        y_ab[n+1] = ab2_step(f_zero, t[n], y_ab[n], y_ab[n-1], h)

    y_am = np.zeros(N+1)
    y_am[0] = 1.5e-4
    y_am[1] = y_am[0]
    for n in range(1, N):
        y_am[n+1] = am2_step(f_zero, t[n], y_am[n], y_am[n-1], h, newton_it=1)

    plt.figure()
    plt.plot(t, np.abs(y_ab), label="AB2")
    plt.plot(t, np.abs(y_am), label="AM2")
    plt.xlabel("t")
    plt.ylabel(r"$|y_n|$")
    plt.title("Stage 8: zero-stability test on $y' = 0$")
    plt.grid(True, ls=":")
    plt.legend()
    plt.savefig("figs/stage8_zero_stability.png", dpi=150, bbox_inches="tight")
    plt.close()

def stage8_convergence_exampleA():
    def f(t, y):
        return -2.0 * y

    def y_exact(t):
        return np.exp(-2.0 * t)

    t0, T = 0.0, 5.0
    y0 = 1.0


    h_vals = np.array([0.2, 0.1, 0.0625, 0.04]) 

    errs_ab2 = []
    errs_am2 = []

    for h in h_vals:
        N = int((T - t0)/h)
        t_grid = np.linspace(t0, T, N+1)
        t_rk, y_rk = rk4(f, t0, t0 + h, y0, h)  
        y1_start = y_rk[-1]

        y_ab = np.zeros(N+1)
        y_ab[0] = y0
        y_ab[1] = y1_start
        y_am = np.zeros(N+1)
        y_am[0] = y0
        y_am[1] = y1_start

        for n in range(1, N):
            y_ab[n+1] = ab2_step(f, t_grid[n], y_ab[n], y_ab[n-1], h)
            y_am[n+1] = am2_step(f, t_grid[n], y_am[n], y_am[n-1], h)

        yT_exact = y_exact(T)
        errs_ab2.append(abs(y_ab[-1] - yT_exact))
        errs_am2.append(abs(y_am[-1] - yT_exact))

    plt.figure()
    plt.loglog(h_vals, errs_ab2, "o-", label="AB2")
    plt.loglog(h_vals, errs_am2, "s-", label="AM2")

    C_ref = errs_ab2[0] / (h_vals[0]**2)
    plt.loglog(h_vals, C_ref * h_vals**2, "k--", label=r"$h^2$ (ref)")

    plt.xlabel("h")
    plt.ylabel("global error at T")
    plt.title("Stage 8: global error vs step size (Example A)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.savefig("figs/stage8_convergence.png", dpi=150, bbox_inches="tight")
    plt.close()

def stage8_starting_values():
    f, t0, T, y0, exact = example_A()
    h = 0.1
    N = int(np.round((T - t0) / h))

    ts = t0 + h * np.arange(N + 1)
    y0_arr = np.array(y0, dtype=float)
    y_good = np.zeros(N + 1)
    y_good[0] = y0_arr
    y_good[1] = rk4_step(f, t0, y0_arr, h)
    f_vals_good = np.zeros(N + 1)
    f_vals_good[0] = f(ts[0], y_good[0])
    f_vals_good[1] = f(ts[1], y_good[1])
    for n in range(1, N):
        y_good[n+1] = y_good[n] + h * (1.5 * f_vals_good[n] - 0.5 * f_vals_good[n-1])
        f_vals_good[n+1] = f(ts[n+1], y_good[n+1])

    y_bad = np.zeros(N + 1)
    y_bad[0] = y0_arr
    y_bad[1] = y0_arr + h * f(t0, y0_arr)
    f_vals_bad = np.zeros(N + 1)
    f_vals_bad[0] = f(ts[0], y_bad[0])
    f_vals_bad[1] = f(ts[1], y_bad[1])
    for n in range(1, N):
        y_bad[n+1] = y_bad[n] + h * (1.5 * f_vals_bad[n] - 0.5 * f_vals_bad[n-1])
        f_vals_bad[n+1] = f(ts[n+1], y_bad[n+1])

    exact_vals = exact(ts)
    err_good = np.abs(y_good - exact_vals)
    err_bad = np.abs(y_bad - exact_vals)

    plt.figure()
    plt.semilogy(ts, err_good, "o-", label="AB2 with RK4 starter (order 4)")
    plt.semilogy(ts, err_bad, "s-", label="AB2 with FE starter (order 1)")
    plt.xlabel("t")
    plt.ylabel("error")
    plt.title("Stage 8: effect of starting values on AB2 (Example A)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.savefig("figs/stage8_starting_values.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Stage 8: starter comparison (Example A, h=0.1)")
    print("starter   final_error")
    print(f"RK4      {err_good[-1]:.3e}")
    print(f"FE       {err_bad[-1]:.3e}")

def run_stage8():
    stage8_zero_stability()
    stage8_convergence_exampleA()
    stage8_starting_values()
