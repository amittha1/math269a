import numpy as np
import matplotlib.pyplot as plt

from .solvers import forward_euler, rk2, rk4
from .problems import example_D, example_E

def time_series(t, y, method, h, fig_path, title_suffix = ""):
    plt.figure()
    plt.plot(t, y[:, 0], label="component 1")
    plt.plot(t, y[:, 1], label="component 2")
    plt.xlabel("t")
    plt.ylabel("solution components")
    title = f"Time Series: {method} (h={h})"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)
    plt.legend()
    plt.grid(True, ls=":")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

def phase(y, method, h, fig_path, title_suffix = ""):
    plt.figure()
    plt.plot(y[:, 0], y[:, 1])
    plt.xlabel("component 1")
    plt.ylabel("component 2")
    title = f"Phase Portrait: {method} (h={h})"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)
    plt.grid(True, ls=":")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

def run_exampleD():
    f, t0, T, y0, A = example_D()

    eigenvalues = np.linalg.eigvals(A)
    min_lam = np.min(np.real(eigenvalues))

    h_max = -2.0 / min_lam
    h_stable = 0.8 * h_max
    h_unstable = 1.2 * h_max
    methods = [
        ("Forward Euler", forward_euler),
        ("RK2", rk2),
        ("RK4", rk4),
    ]

    for method_name, method_func in methods:
        for h, stability in [(h_stable, "stable"), (h_unstable, "unstable")]:
            t, y = method_func(f, t0, T, y0, h)
            time_series(
                t,
                y,
                method_name,
                h,
                fig_path=f"figs/stage4_exampleD_{method_name.replace(' ', '_').lower()}_{stability}.png",
                title_suffix=f"({stability} h)"
            )
            phase(
                y,
                method_name,
                h,
                fig_path=f"figs/stage4_exampleD_phase_{method_name.replace(' ', '_').lower()}_{stability}.png",
                title_suffix=f"({stability} h)"
            )

def run_exampleE():
    f, t0, T, y0, A, exact = example_E()

    h_small = 0.05
    h_large = 0.5

    methods = [
        ("Forward Euler", forward_euler),
        ("RK2", rk2),
        ("RK4", rk4),
    ]

    for method_name, method_func in methods:
        for h, size in [(h_small, "small"), (h_large, "large")]:
            t, y = method_func(f, t0, T, y0, h)
            time_series(
                t,
                y,
                method_name,
                h,
                fig_path=f"figs/stage4_exampleE_{method_name.replace(' ', '_').lower()}_{size}_h.png",
                title_suffix=f"({size} h)"
            )
            phase(
                y,
                method_name,
                h,
                fig_path=f"figs/stage4_exampleE_phase_{method_name.replace(' ', '_').lower()}_{size}_h.png",
                title_suffix=f"({size} h)"
            )
def run_stage4():
    run_exampleD()
    run_exampleE()