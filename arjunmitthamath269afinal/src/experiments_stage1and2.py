import numpy as np
import matplotlib.pyplot as plt

from .solvers import forward_euler, rk2, rk4
from .problems import example_A, example_B


def convergence_exp(problem, method, hs, fig_path, title):
    f, t0, T, y0, exact = problem()

    errors = []
    for h in hs:
        t, y = method(f, t0, T, y0, h)
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

    print(f" {title} ")
    print(f"{'h':>12} {'error':>20} {'E(h)/E(h/2)':>20}")
    for i, h in enumerate(hs):
        ratio = ""
        if i > 0:
            ratio_val = errors[i - 1] / errors[i]
            ratio = f"{ratio_val:>20.6f}"
        print(f"{h:12.6e} {errors[i]:20.6e} {ratio}")
    print()

def work_precision_exp(problem, hs, fig_path, title):
    f, t0, T, y0, exact = problem()

    methods = [
        ("Euler", forward_euler, 1),
        ("RK2", rk2, 2),
        ("RK4", rk4, 4),]
    
    plt.figure()
    for method_name, method_func, order in methods:
        errors = []
        works = []
        for h in hs:
            t, y = method_func(f, t0, T, y0, h)
            err = abs(y[-1] - exact(T))
            N = int((T - t0)/h)
            total_evals = N * order
            works.append(total_evals)
            errors.append(err)

        plt.loglog(works, errors,"o-", label=method_name)

    plt.xlabel("total RHS evaluations")
    plt.ylabel("final-time error")
    plt.title("Work-Precision: " + title)
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()



def run_stage1():
    hs = np.array([2**(-k) for k in range(3, 10)])
    convergence_exp(
        problem = example_A,
        method = forward_euler,
        hs = hs,
        fig_path = "figs/stage1_forward_euler_exampleA.png",
        title = "Stage 1: Forward Euler Convergence (Example A)",
    ) 
    convergence_exp(
        problem = example_B,
        method = forward_euler,
        hs = hs,
        fig_path = "figs/stage1_forward_euler_exampleB.png",
        title  = "Stage 1: Forward Euler Convergence (Example B)",
    )

def run_stage2():
    hs = np.array([2**(-k) for k in range(3, 10)])
    convergence_exp(
        problem = example_A,
        method = rk2,
        hs = hs,
        fig_path = "figs/stage2_rk2_exampleA.png",
        title = "Stage 2: RK2 Convergence (Example A)",
    ) 
    convergence_exp(
        problem = example_B,
        method = rk2,
        hs = hs,
        fig_path = "figs/stage2_rk2_exampleB.png",
        title  = "Stage 2: RK2 Convergence (Example B)",
    )
    convergence_exp(
        problem = example_A,
        method = rk4,
        hs = hs,
        fig_path = "figs/stage2_rk4_exampleA.png",
        title = "Stage 2: RK4 Convergence (Example A)",
    )
    convergence_exp(
        problem = example_B,
        method = rk4,
        hs = hs,
        fig_path = "figs/stage2_rk4_exampleB.png",
        title  = "Stage 2: RK4 Convergence (Example B)",
    )
    work_precision_exp(
        problem = example_A,
        hs = hs,
        fig_path = "figs/stage2_work_precision_exampleA.png",
        title = "Stage 2: Work-Precision (Example A)",
    )

    
