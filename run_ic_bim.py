import numpy as np

from problem import (
    PoissonProblem,
    f_example_2D,
    f_example_3D,
    bc_2D,
    bc_3D,
)


def run_ic_bim_2d():
    """
    Run IC-BIM for the 2D Poisson problem for a range of grid sizes
    and save convergence / timing data to ic_bim_2D.npz.
    """
    # p = 2, ..., 9  -> n = 2^p (same convention as existing 2D runs)
    p_vals = np.arange(2, 10, 1)
    ns = (2 ** p_vals).astype(int)

    hs = []
    e_rms_list = []
    e_inf_list = []
    cpu_fact_list = []
    cpu_iter_list = []
    peak_fact_list = []
    peak_iter_list = []
    niters_list = []
    error_histories = []

    for n in ns:
        print(f"\n===== 2D IC-BIM: n = {n} =====")
        problem = PoissonProblem(n, f_example_2D, bc_2D)
        problem.construct_1d_problem()
        problem.construct_2d_problem()

        # Solve with IC-BIM (factorisation ICU time is stored in problem.ic_cpu_time)
        u, e_rms, e_inf, cpu_iter, peak_iter = problem.ic_bim_solve()

        h = 1.0 / (n + 1)

        hs.append(h)
        e_rms_list.append(e_rms)
        e_inf_list.append(e_inf)
        cpu_fact_list.append(problem.ic_cpu_time)
        cpu_iter_list.append(cpu_iter)
        peak_fact_list.append(problem.ic_peak_memory)
        peak_iter_list.append(peak_iter)

        niters = len(problem.solver_cg.error_history)
        niters_list.append(niters)
        error_histories.append(np.array(problem.solver_cg.error_history))

        print(f"  h = {h:.4e}")
        print(f"  iterations = {niters}")
        print(f"  RMS error = {e_rms:.3e}, inf error = {e_inf:.3e}")
        print(f"  IC factorisation time = {problem.ic_cpu_time:.4e} s, "
              f"iteration time = {cpu_iter:.4e} s")

    # Save as an object array for error histories (varying lengths)
    np.savez(
        "ic_bim_2D.npz",
        p=p_vals,
        n=ns,
        h=np.array(hs),
        e_rms=np.array(e_rms_list),
        e_inf=np.array(e_inf_list),
        cpu_fact=np.array(cpu_fact_list),
        cpu_iter=np.array(cpu_iter_list),
        peak_fact=np.array(peak_fact_list),
        peak_iter=np.array(peak_iter_list),
        niters=np.array(niters_list),
        error_histories=np.array(error_histories, dtype=object),
    )

    print("\nSaved 2D IC-BIM results to ic_bim_2D.npz")


def run_ic_bim_3d():
    """
    Run IC-BIM for the 3D Poisson problem for a range of grid sizes
    and save convergence / timing data to ic_bim_3D.npz.
    """
    # Keep 3D range modest to avoid insane runtimes/memory.
    # p = 2, ..., 6  -> n = 2^p
    p_vals = np.arange(2, 7, 1)
    ns = (2 ** p_vals).astype(int)

    hs = []
    e_rms_list = []
    e_inf_list = []
    cpu_fact_list = []
    cpu_iter_list = []
    peak_fact_list = []
    peak_iter_list = []
    niters_list = []
    error_histories = []

    for n in ns:
        print(f"\n===== 3D IC-BIM: n = {n} =====")
        problem = PoissonProblem(n, f_example_3D, bc_3D)
        problem.construct_1d_problem()
        problem.construct_3d_problem()

        # Solve with IC-BIM
        u, e_rms, e_inf, cpu_iter, peak_iter = problem.ic_bim_solve()

        h = 1.0 / (n + 1)

        hs.append(h)
        e_rms_list.append(e_rms)
        e_inf_list.append(e_inf)
        cpu_fact_list.append(problem.ic_cpu_time)
        cpu_iter_list.append(cpu_iter)
        peak_fact_list.append(problem.ic_peak_memory)
        peak_iter_list.append(peak_iter)

        niters = len(problem.solver_cg.error_history)
        niters_list.append(niters)
        error_histories.append(np.array(problem.solver_cg.error_history))

        print(f"  h = {h:.4e}")
        print(f"  iterations = {niters}")
        print(f"  RMS error = {e_rms:.3e}, inf error = {e_inf:.3e}")
        print(f"  IC factorisation time = {problem.ic_cpu_time:.4e} s, "
              f"iteration time = {cpu_iter:.4e} s")

    np.savez(
        "ic_bim_3D.npz",
        p=p_vals,
        n=ns,
        h=np.array(hs),
        e_rms=np.array(e_rms_list),
        e_inf=np.array(e_inf_list),
        cpu_fact=np.array(cpu_fact_list),
        cpu_iter=np.array(cpu_iter_list),
        peak_fact=np.array(peak_fact_list),
        peak_iter=np.array(peak_iter_list),
        niters=np.array(niters_list),
        error_histories=np.array(error_histories, dtype=object),
    )

    print("\nSaved 3D IC-BIM results to ic_bim_3D.npz")


if __name__ == "__main__":
    # You can comment out one of these if you only want 2D or 3D.
    #run_ic_bim_2d()
    run_ic_bim_3d()
