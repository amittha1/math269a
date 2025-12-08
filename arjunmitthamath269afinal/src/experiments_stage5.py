import numpy as np
import matplotlib.pyplot as plt

from .solvers import rk4, backward_euler, trapezoidal
from .problems import example_A_s5, example_B_s5, example_F



def convergence_implicit_example(example_fn, name, fig_path):
    f, df_dy, t0, T, y0, exact = example_fn()
    hs = [2.0 ** (-k) for k in range(3, 10)]  
    errors_be = []
    errors_tr = []

    C_be = 0.1   
    C_tr = 0.1   
    for h in hs:
        tol_be = C_be * h ** 2
        t_be, y_be, it_be = backward_euler(f, df_dy, t0, T, y0, h, tol_be)
        err_be = abs(y_be[-1] - exact(T))
        errors_be.append(err_be)

        tol_tr = C_tr * h ** 3
        t_tr, y_tr, it_tr = trapezoidal(f, df_dy, t0, T, y0, h, tol_tr)
        err_tr = abs(y_tr[-1] - exact(T))
        errors_tr.append(err_tr)

    plt.figure()
    plt.loglog(hs, errors_be, "o-", label="Backward Euler (p=1)")
    plt.loglog(hs, errors_tr, "s-", label="Trapezoidal (p=2)")

    h_ref = np.array(hs)
    plt.loglog(h_ref, h_ref, "--", label="h^1 (ref)")
    plt.loglog(h_ref, h_ref ** 2, "--", label="h^2 (ref)")

    plt.gca().invert_xaxis()  
    plt.xlabel("h")
    plt.ylabel("final-time error")
    plt.title(f"Stage 5: Implicit convergence on {name}")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def reference_rk4_exampleF():
    f, df_dy, t0, T, y0, _ = example_F()
    h_ref = 2.0 ** (-12)
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


def newton_stats_exampleF(fig_path=None):
    f, df_dy, t0, T, y0, _ = example_F()
    t_ref, y_ref = reference_rk4_exampleF()

    def interp_ref(t):
        return np.interp(t, t_ref, y_ref)

    hs = [2.0 ** (-k) for k in range(3, 8)]  
    C_be = 0.1
    C_tr = 0.1

    stats = []

    for h in hs:
        tol_be = C_be * h ** 2
        tol_tr = C_tr * h ** 3

        
        t_be, y_be, it_be = backward_euler(f, df_dy, t0, T, y0, h, tol_be)
        err_be = abs(y_be[-1] - interp_ref(T))
        min_be = int(np.min(it_be))
        mean_be = float(np.mean(it_be))
        max_be = int(np.max(it_be))

        
        t_tr, y_tr, it_tr = trapezoidal(f, df_dy, t0, T, y0, h, tol_tr)
        err_tr = abs(y_tr[-1] - interp_ref(T))
        min_tr = int(np.min(it_tr))
        mean_tr = float(np.mean(it_tr))
        max_tr = int(np.max(it_tr))

        stats.append(
            (h, err_be, min_be, mean_be, max_be, err_tr, min_tr, mean_tr, max_tr)
        )

    
    print("Example F: Newton iteration statistics")
    print("h        | BE_err   BE_it(min/mean/max)   | TR_err   TR_it(min/mean/max)")
    for (
        h,
        err_be,
        min_be,
        mean_be,
        max_be,
        err_tr,
        min_tr,
        mean_tr,
        max_tr,
    ) in stats:
        print(
            f"{h:8.5f}  | {err_be:8.2e}  "
            f"{min_be:2d}/{mean_be:4.1f}/{max_be:2d}    | {err_tr:8.2e}  "
            f"{min_tr:2d}/{mean_tr:4.1f}/{max_tr:2d}"
        )



def explicit_vs_implicit_exampleF(fig_path):
    f, df_dy, t0, T, y0, _ = example_F()
    t_ref, y_ref = reference_rk4_exampleF()

    def interp_ref(t):
        return np.interp(t, t_ref, y_ref)

    h_exp = 2.0 ** (-8)
    t_rk4, y_rk4 = rk4(f, t0, T, y0, h_exp)

    
    h_imp = 2.0 ** (-3) 
    C_be = 0.1
    C_tr = 0.1
    tol_be = C_be * h_imp ** 2
    tol_tr = C_tr * h_imp ** 3

    t_be, y_be, it_be = backward_euler(f, df_dy, t0, T, y0, h_imp, tol_be)
    t_tr, y_tr, it_tr = trapezoidal(f, df_dy, t0, T, y0, h_imp, tol_tr)

    
    yT_ref = interp_ref(T)
    err_rk4 = abs(y_rk4[-1] - yT_ref)
    err_be = abs(y_be[-1] - yT_ref)
    err_tr = abs(y_tr[-1] - yT_ref)

    print("\nExample F: explicit vs implicit at a target accuracy")
    print(f"RK4 (h={h_exp}):    final error ~ {err_rk4:8.2e}, steps = {len(t_rk4)-1}")
    print(f"BE  (h={h_imp}):    final error ~ {err_be:8.2e}, steps = {len(t_be)-1}")
    print(f"Trap(h={h_imp}):    final error ~ {err_tr:8.2e}, steps = {len(t_tr)-1}")

    
    plt.figure()
    plt.plot(t_ref, y_ref, "-", label="reference (fine RK4)")
    plt.plot(t_rk4, y_rk4, "--", label=f"RK4, h={h_exp}")
    plt.plot(t_be, y_be, "o-", label=f"Backward Euler, h={h_imp}")
    plt.plot(t_tr, y_tr, "s-", label=f"Trapezoidal, h={h_imp}")
    plt.xlabel("t")
    plt.ylabel("y(t)")
    plt.title("Stage 5: Example F, explicit vs implicit")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_stage5():
    convergence_implicit_example(
        example_A_s5, "Example A", "figs/stage5_convergence_exampleA.png"
    )
    convergence_implicit_example(
        example_B_s5, "Example B", "figs/stage5_convergence_exampleB.png"
    )
    newton_stats_exampleF("figs/stage5_exampleF_implicit_error.png")
    explicit_vs_implicit_exampleF("figs/stage5_exampleF_explicit_vs_implicit.png")
