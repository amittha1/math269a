import numpy as np
import matplotlib.pyplot as plt

from .problems import test_linear_problem, example_C

def euler_step(f, t, y, h):
    return t + h, y + h * f(t, y)


def rk2_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return t + h, y + 0.5 * h * (k1 + k2)

def stability_exp(stepper, lambda_val, method, h_stable, h_unstable, N_steps, t0, T, y0, fig_path, title):
    plt.figure()
    f, t0, _, y0, _ = test_linear_problem(lambda_val, t0, T, y0)
    for h, label_suffix, linestyle in [
        (h_stable, "stable", "o-"),
        (h_unstable, "unstable", "s--")
    ]:
        t = t0
        y = y0
        magnitudes = [abs(y)]
        indices = [0]
        for n in range(1, N_steps + 1):
            t, y = stepper(f, t, y, h)
            magnitudes.append(abs(y))
            indices.append(n)
        plt.semilogy(indices, magnitudes, linestyle, label=f"h={h} ({label_suffix})")
    plt.xlabel("Step index n")
    plt.ylabel(r"$|y_n|$")
    plt.title(f"{title}: {method}, (lambda={lambda_val})")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

def run_stability_s3():
    t0 = 0.0
    T = 5.0
    y0 = 1.0
    N_steps = 50

    
    for lam in [-1.0, -10]:
        h_max = 2.0 / abs(lam)
        h_stable_euler = 0.8 * h_max
        h_unstable_euler = 1.2 * h_max
        stability_exp(
            stepper=euler_step,
            lambda_val=lam,
            method="Forward Euler",
            h_stable=h_stable_euler,
            h_unstable=h_unstable_euler,
            N_steps=N_steps,
            t0=t0,
            T=T,
            y0=y0,
            fig_path=f"figs/stage3_euler_lambda{int(lam)}.png",
            title="Stage 3 Stability"
        )

    for lam in [-1.0, -10.0]:
        h_max = 2.0 / abs(lam)
        h_stable_rk2 = 0.8 * h_max
        h_unstable_rk2 = 1.2 * h_max
        stability_exp(
            stepper=rk2_step,
            lambda_val=lam,
            method="RK2",
            h_stable=h_stable_rk2,
            h_unstable=h_unstable_rk2,
            N_steps=N_steps,
            t0=t0,
            T=T,
            y0=y0,
            fig_path=f"figs/stage3_rk2_lambda{int(lam)}.png",
            title="Stability Experiment"
        )
    

def reference_exampleC():
    f, df_dy, t0, T, y0 = example_C()
    h_ref = 0.001
    N = int((T - t0) / h_ref)
    
    t_vals = t0 + h_ref * np.arange(N + 1)
    y_vals = np.zeros_like(t_vals)
    y_vals[0] = y0
    for n in range(N):
        t = t_vals[n]
        y = y_vals[n]
        k1 = f(t, y)
        k2  = f(t + 0.5 * h_ref, y + 0.5 * h_ref * k1)
        k3 = f(t + 0.5 * h_ref, y + 0.5 * h_ref * k2)
        k4 = f(t + h_ref, y + h_ref * k3)
        y_vals[n] = y + (h_ref / 6) * (k1 + 2 * k2 + 2 * k3 + k4)



    lambdas = df_dy(t_vals, y_vals)
    lambda_min = np.min(lambdas)
    return t_vals, lambdas, lambda_min

def run_exampleC_stability():
    f , df_dy, t0, T, y0 = example_C()
    t_vals, lambda_vals, lambda_min = reference_exampleC()
    
    h_max = 2.0 / abs(lambda_min)

    h_euler_stable   = 0.5 * h_max
    h_euler_unstable = 2.5 * h_max

    # plot h * df/dy for the unstable h and mark the -2 threshold
    z_unstable = h_euler_unstable * lambda_vals  # z(t) = h * lambda(t)
    z_stable   = h_euler_stable * lambda_vals

    plt.figure()
    plt.plot(t_vals, z_unstable, label=rf"$z(t) = h \,\partial f/\partial y$ with $h={h_euler_unstable:.4f}$")
    plt.plot(t_vals, z_stable, label=rf"$z(t) = h \,\partial f/\partial y$ with $h={h_euler_stable:.4f}$")
    plt.axhline(-2.0, color="k", linestyle="--", label="stability boundary $z=-2$")
    plt.xlabel("t")
    plt.ylabel(r"$h\,\partial f/\partial y(t,y(t))$")
    plt.title("Example C: stability indicator for Forward Euler")
    plt.grid(True, ls=":")
    plt.legend()
    plt.savefig("figs/stage3_exampleC_euler_indicator.png", dpi=150, bbox_inches="tight")
    plt.close()

    # same idea for RK2; real-axis bound is the same, so the picture is identical in theory,
    # but we still show it for completeness if you want a second plot:
    h_rk2_stable   = 0.5 * h_max
    h_rk2_unstable = 2.5 * h_max
    z_unstable_rk2 = h_rk2_unstable * lambda_vals
    z_stable_rk2   = h_rk2_stable * lambda_vals

    plt.figure()
    plt.plot(t_vals, z_unstable_rk2, label=rf"$z(t) = h \,\partial f/\partial y$ with $h={h_rk2_unstable:.4f}$")
    plt.plot(t_vals, z_stable_rk2, label=rf"$z(t) = h \,\partial f/\partial y$ with $h={h_rk2_stable:.4f}$")
    plt.axhline(-2.0, color="k", linestyle="--", label="stability boundary $z=-2$")
    plt.xlabel("t")
    plt.ylabel(r"$h\,\partial f/\partial y(t,y(t))$")
    plt.title("Example C: stability indicator for RK2")
    plt.grid(True, ls=":")
    plt.legend()
    plt.savefig("figs/stage3_exampleC_rk2_indicator.png", dpi=150, bbox_inches="tight")
    plt.close()

    return h_max, lambda_min

def run_stage3():
    run_stability_s3()
    run_exampleC_stability()
