import numpy as np
import matplotlib.pyplot as plt

from .problems import example_G
from .solvers import forward_euler, rk2, rk4, backward_euler, trapezoidal

def run_method(method, f, df_dy, t0, T, y0, h, exact=None, implicit=False):

    if implicit:
        if method == backward_euler:
            tol = 0.1 * h**2
            t, y, iters = method(f, df_dy, t0, T, y0, h, tol)
            work = len(t)-1 
        else:
            tol = 0.1 * h**3
            t, y, iters = method(f, df_dy, t0, T, y0, h, tol)
            work = len(t)-1
    else:
        t, y = method(f, t0, T, y0, h)
        if method == forward_euler:
            work = len(t)-1
        elif method == rk2:
            work = 2*(len(t)-1)
        elif method == rk4:
            work = 4*(len(t)-1)
    
    err = abs(y[-1] - exact(T)) 

    return err, work


def stability_test_example_G(fig_path_prefix):
    f, df_dy, t0, T, y0, exact = example_G()
    a_FE = -2.0
    lambda_min = -1000.0
    h_sigma_FE = abs(a_FE / lambda_min) 
    h_stable = 0.8 * h_sigma_FE
    h_unstable = 1.2 * h_sigma_FE

    for h, tag in [(h_stable, "stable"), (h_unstable, "unstable")]:
        t, y = forward_euler(f, t0, T, y0, h)
        plt.figure()
        plt.plot(t, y)
        plt.xlabel("t")
        plt.ylabel("y(t)")
        plt.title(f"Example G — Forward Euler, h={h:.4e} ({tag})")
        plt.grid(True)
        plt.savefig(f"{fig_path_prefix}_FE_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"Example G: lambda_min={lambda_min}, h_sigma_FE ≈ {h_sigma_FE}")

def estimate_h_tau(method, f, df_dy, t0, T, y0, exact, method_name, implicit=False, tol=1e-4):

    hs = [2.0**(-k) for k in range(2,12)]  
    acceptable = []

    for h in hs:
        err, work = run_method(method, f, df_dy, t0, T, y0, h, exact, implicit)
        if err <= tol:
            acceptable.append((h, work, err))

    if len(acceptable)==0:
        return None
    h_tau, work_tau, err_tau = acceptable[0]
    return h_tau, work_tau, err_tau


def work_precision_s6(fig_path):
    f, df_dy, t0, T, y0, exact = example_G()
    tol = 1e-4

    results = {}

    results["Forward Euler"] = estimate_h_tau(forward_euler, f, df_dy, t0, T, y0, exact,
                                             "FE", implicit=False, tol=tol)
    results["RK2"] = estimate_h_tau(rk2, f, df_dy, t0, T, y0, exact,
                                   "RK2", implicit=False, tol=tol)
    results["RK4"] = estimate_h_tau(rk4, f, df_dy, t0, T, y0, exact,
                                   "RK4", implicit=False, tol=tol)

    results["Backward Euler"] = estimate_h_tau(backward_euler, f, df_dy, t0, T, y0, exact,
                                              "BE", implicit=True, tol=tol)
    results["Trapezoidal"] = estimate_h_tau(trapezoidal, f, df_dy, t0, T, y0, exact,
                                           "TR", implicit=True, tol=tol)

    print("\nStage 6: h_tau and work at tol =", tol)
    for k,v in results.items():
        print(k, "----", v)

    labels = []
    works = []
    for k,v in results.items():
        if v is not None:
            labels.append(k)
            works.append(v[1])

    plt.figure()
    plt.bar(labels, works)
    plt.yscale("log")
    plt.ylabel("Work (total RHS evals in log scale)")
    plt.title("Stage 6: Work to achieve tol = 1e-4 (Example G)")
    plt.grid(axis="y")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

def run_stage6():
    stability_test_example_G("figs/stage6_stability")
    work_precision_s6("figs/stage6_work_precision.png")
