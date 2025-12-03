import numpy as np
import matplotlib.pyplot as plt

from .solvers import forward_euler
from .problems import example_A, example_B


def stage_1(problem, hs, fig_path, title):
    f, t0, T, y0, exact = problem()

    errors = []
    for h in hs:
        t, y = forward_euler(f, t0, T, y0, h)
        err = abs(y[-1] - exact(T))
        errors.append(err)

    errors = np.array(errors)
    plt.figure()
    plt.loglog(hs, errors, "o-")
    plt.xlabel("h")
    plt.ylabel("final-time error")
    plt.title(title)
    plt.grid(True, which="both", ls=":")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # print table with error ratios
    print(f"\n=== {title} ===")
    print(f"{'h':>12} {'error':>20} {'E(h)/E(h/2)':>20}")
    print("-" * 60)
    for i, h in enumerate(hs):
        ratio = ""
        if i > 0:
            ratio_val = errors[i - 1] / errors[i]
            ratio = f"{ratio_val:>20.6f}"
        print(f"{h:12.6e} {errors[i]:20.6e} {ratio}")
    print()
def run_stage1():
    hs = np.array([2**(-k) for k in range(3, 10)])
    stage_1(
        problem = example_A,
        hs = hs,
        fig_path = "figs/stage1_forward_euler_exampleA.png",
        title = "Stage 1: Forward Euler Convergence (Example A)",
    ) 
    stage_1(
        problem = example_B,
        hs = hs,
        fig_path = "figs/stage1_forward_euler_exampleB.png",
        title  = "Stage 1: Forward Euler Convergence (Example B)",
    )

    
