import numpy as np
import matplotlib.pyplot as plt

from .problems import example_A_s5, example_H
from .solvers import rk4



def rk2_step(f, t, y, h):

    y = np.asarray(y, dtype=float)
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    y_next = y + 0.5 * h * (k1 + k2)
    return y_next


def err_inf_norm(err_vec):
    arr = np.asarray(err_vec, dtype=float)
    if arr.ndim == 0:
        return float(abs(arr))
    return float(np.max(np.abs(arr)))


def adaptive_rk2_stepdoubling(
    f, t0, T, y0, h0, tol_loc,
    p=2, safety=0.9, h_min=1e-8, h_max=None, max_steps=100000
):

    if h_max is None:
        h_max = T - t0

    t = float(t0)
    y = np.asarray(y0, dtype=float)

    ts = [t]
    ys = [y.copy()]
    hs = []
    err_local = []

    n_rhs = 0
    n_accept = 0
    n_reject = 0
    h = h0
    n_steps = 0

    while t < T and n_steps < max_steps:
        h = min(h, h_max, T - t)
        if h < h_min * 0.999:
            break

        y_big = rk2_step(f, t, y, h)

        y_half1 = rk2_step(f, t, y, 0.5 * h)
        y_half2 = rk2_step(f, t + 0.5 * h, y_half1, 0.5 * h)

        n_rhs += 3 * 2

        err_vec = y_half2 - y_big
        err = err_inf_norm(err_vec)

        if err == 0.0:
            factor = 2.0
        else:
            factor = safety * (tol_loc / err) ** (1.0 / (p + 1))
        factor = max(0.2, min(5.0, factor))
        h_new = max(h_min, min(h_max, factor * h))
        if (err <= tol_loc) or (h <= h_min * 1.001):
            t += h
            y = y_half2
            ts.append(t)
            ys.append(y.copy())
            hs.append(h)
            err_local.append(err)
            n_accept += 1
            h = h_new
        else:
            n_reject += 1
            h = h_new

        n_steps += 1

    ts = np.array(ts)
    ys = np.vstack(ys)
    hs = np.array(hs)
    err_local = np.array(err_local)

    if ys.ndim == 2 and ys.shape[1] == 1:
        ys = ys[:, 0]

    return ts, ys, hs, err_local, n_rhs, n_accept, n_reject


def rk23_step(f, t, y, h):
    y = np.asarray(y, dtype=float)

    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.75 * h, y + 0.75 * h * k2)
    y3 = y + h * (2.0 / 9.0 * k1 + 1.0 / 3.0 * k2 + 4.0 / 9.0 * k3)  # 3rd order

    k4 = f(t + h, y3)
    y2 = y + h * (7.0 / 24.0 * k1 + 0.25 * k2 + 1.0 / 3.0 * k3 + 1.0 / 8.0 * k4)  # 2nd

    err_vec = y3 - y2
    return y3, err_vec, 4

def adaptive_rk23(
    f, t0, T, y0, h0, tol_loc,
    p=2, safety=0.9, h_min=1e-8, h_max=None, max_steps=100000
):
   
    if h_max is None:
        h_max = T - t0

    t = float(t0)
    y = np.asarray(y0, dtype=float)

    ts = [t]
    ys = [y.copy()]
    hs = []
    err_local = []

    n_rhs = 0
    n_accept = 0
    n_reject = 0
    h = h0
    n_steps = 0

    while t < T and n_steps < max_steps:
        h = min(h, h_max, T - t)
        if h < h_min * 0.999:
            break

        y_new, err_vec, n_rhs_step = rk23_step(f, t, y, h)
        n_rhs += n_rhs_step

        err = err_inf_norm(err_vec)

        if err == 0.0:
            factor = 2.0
        else:
            factor = safety * (tol_loc / err) ** (1.0 / (p + 1))
        factor = max(0.2, min(5.0, factor))
        h_new = max(h_min, min(h_max, factor * h))

        if (err <= tol_loc) or (h <= h_min * 1.001):
            t += h
            y = y_new
            ts.append(t)
            ys.append(y.copy())
            hs.append(h)
            err_local.append(err)
            n_accept += 1
            h = h_new
        else:
            n_reject += 1
            h = h_new

        n_steps += 1

    ts = np.array(ts)
    ys = np.vstack(ys)
    hs = np.array(hs)
    err_local = np.array(err_local)

    if ys.ndim == 2 and ys.shape[1] == 1:
        ys = ys[:, 0]

    return ts, ys, hs, err_local, n_rhs, n_accept, n_reject


def reference_solution(f, t0, T, y0, h_ref=1e-3):
    N = int((T - t0) / h_ref)
    ts = np.linspace(t0, T, N + 1)
    ys = np.zeros(N + 1)
    ys[0] = y0

    for n in range(N):
        t = ts[n]
        y = ys[n]
        k1 = f(t, y)
        k2 = f(t + 0.5 * h_ref, y + 0.5 * h_ref * k1)
        k3 = f(t + 0.5 * h_ref, y + 0.5 * h_ref * k2)
        k4 = f(t + h_ref, y + h_ref * k3)
        ys[n + 1] = y + (h_ref / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ts, ys
def reference_solution_H(f, t0, T, y0, h_ref=1e-3):
    t0 = float(t0)
    T = float(T)
    y0 = np.asarray(y0, dtype=float)

    N = int(np.round((T - t0) / h_ref))
    h_ref = (T - t0) / N  

    ts = np.linspace(t0, T, N + 1)

    if y0.ndim == 0:
        ys = np.zeros(N + 1, dtype=float)
    else:
        d = y0.size
        ys = np.zeros((N + 1, d), dtype=float)

    y = y0.copy()
    ys[0] = y0

    t = t0
    for n in range(N):
        k1 = f(t, y)
        k2 = f(t + 0.5 * h_ref, y + 0.5 * h_ref * k1)
        k3 = f(t + 0.5 * h_ref, y + 0.5 * h_ref * k2)
        k4 = f(t + h_ref,       y + h_ref * k3)

        y = y + (h_ref / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t = ts[n+1]
        ys[n+1] = y

   
    if ys.ndim == 2 and ys.shape[1] == 1:
        ys = ys[:, 0]

    return ts, ys

def stage7_exampleA_stepdoubling():
    f, df_dy, t0, T, y0, exact = example_A_s5()
    tol = 1e-4
    h0 = (T - t0) / 10.0
    ts, ys, hs, errs, n_rhs, n_acc, n_rej = adaptive_rk2_stepdoubling(
        f, t0, T, y0, h0, tol_loc=tol
    )

    t_ref, y_ref = reference_solution_H(f, t0, T, y0, h_ref=2.0 ** -11)
    if ys.ndim == 2 and ys.shape[1] == 1:
        ys_plot = ys[:, 0]
    else:
        ys_plot = ys

    plt.figure()
    plt.plot(t_ref, y_ref, "k-", label="reference (RK4 fine)")
    plt.plot(ts, ys_plot, "o-", label="adaptive RK2 (step-doubling)")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Stage 7, Example A: adaptive RK2 vs reference")
    plt.grid(True, ls=":")
    plt.legend()
    plt.savefig("figs/stage7_exampleA_solution.png", dpi=150, bbox_inches="tight")
    plt.close()

    t_mid = 0.5 * (ts[:-1] + ts[1:])
    plt.figure()
    plt.plot(t_mid, hs, "o-")
    plt.xlabel("t")
    plt.ylabel("h")
    plt.title("Stage 7, Example A: adaptive RK2 stepsizes")
    plt.grid(True, ls=":")
    plt.savefig("figs/stage7_exampleA_stepsizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    
    plt.figure()
    plt.hist(errs, bins=30)
    plt.xlabel("local error estimate")
    plt.ylabel("count")
    plt.title("Stage 7, Example A: error histogram (step-doubling)")
    plt.yscale("log")
    plt.grid(True, ls=":")
    plt.savefig("figs/stage7_exampleA_error_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    final_err = abs(ys_plot[-1] - exact(T))
    print("Stage 7 Example A (RK2 step-doubling):")
    print(f"  tol = {tol:g}")
    print(f"  final error = {final_err:.3e}")
    print(f"  total RHS evals = {n_rhs}")
    print(f"  accepts = {n_acc}, rejects = {n_rej}")



def stage7_exampleH_stepdoubling(mu, tag):
    f, t0, T, y0 = example_H(mu=mu)

    t_ref, y_ref = reference_solution_H(f, t0, T, y0, h_ref=1e-3)

    tol = 1e-4
    h0 = 0.05

    ts, ys, hs, errs, n_rhs, n_acc, n_rej = adaptive_rk2_stepdoubling(
        f, t0, T, y0, h0, tol_loc=tol,
        h_min=1e-6, h_max=0.5
    )

    x_ref = y_ref[:, 0]
    x_ad = ys[:, 0]

    plt.figure()
    plt.plot(t_ref, x_ref, "k-", label="reference (RK4)")
    plt.plot(ts, x_ad, "o-", label="adaptive RK2")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title(f"Stage 7, Example H (mu={mu}): adaptive RK2 vs reference")
    plt.grid(True, ls=":")
    plt.legend()
    plt.savefig(f"figs/stage7_exampleH_mu{tag}_solution.png", dpi=150, bbox_inches="tight")
    plt.close()

    t_mid = 0.5 * (ts[:-1] + ts[1:])
    plt.figure()
    plt.plot(t_mid, hs, "o-")
    plt.xlabel("t")
    plt.ylabel("h")
    plt.title(f"Stage 7, Example H (mu={mu}): adaptive RK2 stepsizes")
    plt.grid(True, ls=":")
    plt.savefig(f"figs/stage7_exampleH_mu{tag}_stepsizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(errs, bins=30)
    plt.xlabel("local error estimate")
    plt.ylabel("count")
    plt.title(f"Stage 7, Example H (mu={mu}): error histogram")
    plt.yscale("log")
    plt.grid(True, ls=":")
    plt.savefig(f"figs/stage7_exampleH_mu{tag}_error_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    final_err = err_inf_norm(ys[-1] - y_ref[-1])
    print(f"Stage 7 Example H (mu={mu}, RK2 step-doubling):")
    print(f"  tol = {tol:g}")
    print(f"  final error = {final_err:.3e}")
    print(f"  total RHS evals = {n_rhs}")
    print(f"  accepts = {n_acc}, rejects = {n_rej}")


def stage7_exampleH_error_vs_tol(mu=10.0):
    f, t0, T, y0 = example_H(mu=mu)
    t_ref, y_ref = reference_solution_H(f, t0, T, y0, h_ref=1e-3)

    tols = [1e-2, 1e-3, 1e-4, 1e-5]
    errors = []
    works = []

    for tol in tols:
        ts, ys, hs, errs, n_rhs, n_acc, n_rej = adaptive_rk2_stepdoubling(
            f, t0, T, y0, h0=0.05, tol_loc=tol,
            h_min=1e-6, h_max=0.5
        )
        err_final = err_inf_norm(ys[-1] - y_ref[-1])
        errors.append(err_final)
        works.append(n_rhs)
        print(f"Example H (mu={mu}), tol={tol:g}: final error={err_final:.3e}, work={n_rhs}")

    plt.figure()
    plt.loglog(tols, errors, "o-", label="adaptive RK2")
    plt.loglog(tols, tols, "k--", label="y = tol")
    plt.xlabel("requested tolerance")
    plt.ylabel("final global error (inf-norm)")
    plt.title(f"Stage 7, Example H (mu={mu}): error vs tolerance")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.savefig(f"figs/stage7_exampleH_mu{int(mu)}_error_vs_tol.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.loglog(tols, works, "o-")
    plt.xlabel("requested tolerance")
    plt.ylabel("work (RHS evaluations)")
    plt.title(f"Stage 7, Example H (mu={mu}): work vs tolerance")
    plt.grid(True, which="both", ls=":")
    plt.gca().invert_xaxis()
    plt.savefig(f"figs/stage7_exampleH_mu{int(mu)}_work_vs_tol.png",
                dpi=150, bbox_inches="tight")
    plt.close()

def stage7_exampleA_rk23():
    f, df_dy, t0, T, y0, exact = example_A_s5()

    tol = 1e-4
    h0 = (T - t0) / 10.0

    ts, ys, hs, errs, n_rhs, n_acc, n_rej = adaptive_rk23(
        f, t0, T, y0, h0, tol_loc=tol
    )

    t_ref, y_ref = reference_solution(f, t0, T, y0, h_ref=2.0 ** -11)

    if ys.ndim == 2 and ys.shape[1] == 1:
        ys_plot = ys[:, 0]
    else:
        ys_plot = ys

    plt.figure()
    plt.plot(t_ref, y_ref, "k-", label="reference (RK4 fine)")
    plt.plot(ts, ys_plot, "o-", label="adaptive RK23")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Stage 7, Example A: adaptive RK23 vs reference")
    plt.grid(True, ls=":")
    plt.legend()
    plt.savefig("figs/stage7_exampleA_rk23_solution.png", dpi=150, bbox_inches="tight")
    plt.close()

    t_mid = 0.5 * (ts[:-1] + ts[1:])
    plt.figure()
    plt.plot(t_mid, hs, "o-")
    plt.xlabel("t")
    plt.ylabel("h")
    plt.title("Stage 7, Example A: adaptive RK23 stepsizes")
    plt.grid(True, ls=":")
    plt.savefig("figs/stage7_exampleA_rk23_stepsizes.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(errs, bins=30)
    plt.xlabel("local error estimate")
    plt.ylabel("count")
    plt.title("Stage 7, Example A: error histogram (RK23)")
    plt.yscale("log")
    plt.grid(True, ls=":")
    plt.savefig("figs/stage7_exampleA_rk23_error_hist.png", dpi=150, bbox_inches="tight")
    plt.close()

    final_err = abs(ys_plot[-1] - exact(T))
    print("Stage 7 Example A (RK23):")
    print(f"  tol = {tol:g}")
    print(f"  final error = {final_err:.3e}")
    print(f"  total RHS evals = {n_rhs}")
    print(f"  accepts = {n_acc}, rejects = {n_rej}")


def run_stage7():
    stage7_exampleA_stepdoubling()
    stage7_exampleH_stepdoubling(mu=1.0, tag="1")
    stage7_exampleH_stepdoubling(mu=10.0, tag="10")
    stage7_exampleH_error_vs_tol(mu=10.0)
    stage7_exampleA_rk23()
