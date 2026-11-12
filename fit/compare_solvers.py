"""The C++ core against the numpy reference on the 100 real problems, and how long each takes (SPEC.md 8).

    python fit/compare_solvers.py [--repeat 5] [--label NAME]

Each portrait is fitted on all its valid landmarks with the final setting and landmark offsets of run_fit.py,
by both solvers, from the same start. Writes fit/results/solver_agreement.json: the largest differences between the two solutions
and the median time per solve on this machine. The C++ solve runs on one thread and the batch on every core;
numpy runs as installed, which usually means a tuned, multi-threaded BLAS library under its matrix product.
With --label NAME the file is solver_agreement_NAME.json, for a second build of the C++ module
(FITCORE_BUILD_DIR selects the build).
"""
import argparse
import json
import platform
import time

import numpy as np

import problem
import run_fit
import solver_ref


def numpy_blas():
    """The name of the BLAS library numpy was built with, or "unknown"."""
    try:
        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=5, help="timing repeats; the fastest pass is kept")
    ap.add_argument("--label", default="", help="name of this C++ build, added to the output file name")
    a = ap.parse_args()
    core = problem.load_fitcore()
    if core is None:
        raise SystemExit("the fitcore module is not built (see fit/fitcore/README.md)")

    setting = json.loads((problem.RESULTS / "summary.json").read_text())["final_fit"]["setting"]
    F = run_fit.Fitter("ref")
    N = len(F.ids)
    n_shape, n_expr = setting["n_shape"], setting["n_expr"]
    f = problem.focal_px(setting["focal_mm"], F.W, F.H)
    ws = np.where(F.valid[None, :], 100.0 / F.iod[:, None], 0.0)
    B = F.basis(n_shape, n_expr)
    p = np.r_[np.full(n_shape, setting["p_shape"]), np.full(n_expr, setting["p_expr"])]
    offsets = [(np.arange(N), np.load(problem.RESULTS / "landmark_offsets.npz")["learned_from_all"])]
    P0 = F.groups(offsets)[0][1]

    # The shared start: the mean-face pose from the reference, coefficients at zero.
    _, _, pose = F.fit(np.ones(problem.N_LANDMARKS, bool), focal_mm=setting["focal_mm"], offsets=offsets)
    Rs, ts = np.stack([r["R"] for r in pose]), np.stack([r["t"] for r in pose])
    thetas = np.zeros((N, n_shape + n_expr))
    model = (P0, B)
    camera = (p, f, F.cx, F.cy)
    opts = problem.SOLVE_OPTIONS

    def one(solver, i):
        return solver.solve(*model, F.xy[i], ws[i], *camera, Rs[i], ts[i], thetas[i], **opts)

    def timed(solver):
        """Results and the per-solve times of the fastest of a.repeat passes over the 100 problems."""
        best, results = None, None
        for _ in range(a.repeat):
            times, results = [], []
            for i in range(N):
                t0 = time.perf_counter()
                results.append(one(solver, i))
                times.append(time.perf_counter() - t0)
            if best is None or sum(times) < sum(best):
                best = times
        return results, np.array(best)

    ref, t_ref = timed(solver_ref)
    cpp, t_cpp = timed(core)
    diff = {k: max(float(np.max(np.abs(ref[i][k] - cpp[i][k]))) for i in range(N)) for k in ("R", "t", "theta")}
    cost_rel = max(abs(ref[i]["cost"] - cpp[i]["cost"]) / ref[i]["cost"] for i in range(N))
    same_status = sum(ref[i]["status"] == cpp[i]["status"] for i in range(N))
    same_iterations = sum(ref[i]["iterations"] == cpp[i]["iterations"] for i in range(N))

    batch = {}
    for label, n_threads in (("one_thread", 1), ("all_threads", 0)):
        best = np.inf
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            out = core.solve_batch(*model, F.xy, ws, *camera, Rs, ts, thetas, n_threads=n_threads, **opts)
            best = min(best, time.perf_counter() - t0)
        assert all(np.array_equal(out[i]["theta"], cpp[i]["theta"]) for i in range(N))
        batch[label] = round(best * 1000.0, 2)

    report = dict(
        problems=N, landmarks_used=int(F.valid.sum()), coefficients=n_shape + n_expr, setting=setting,
        options=opts,
        largest_difference=dict(R=diff["R"], t_metres=diff["t"], theta=diff["theta"], cost_relative=cost_rel),
        same_status=same_status, same_iteration_count=same_iterations,
        iterations_median=float(np.median([r["iterations"] for r in ref])),
        not_converged=dict(numpy=sum(r["status"] != "converged" for r in ref),
                           cpp=sum(r["status"] != "converged" for r in cpp)),
        ms_per_solve_median=dict(numpy=round(float(np.median(t_ref)) * 1000.0, 3),
                                 cpp=round(float(np.median(t_cpp)) * 1000.0, 3)),
        speedup_median=round(float(np.median(t_ref) / np.median(t_cpp)), 1),
        loop_ms=dict(numpy=round(float(t_ref.sum()) * 1000.0, 1), cpp=round(float(t_cpp.sum()) * 1000.0, 1)),
        batch_100_ms=batch,
        machine=dict(processor=platform.processor() or platform.machine(), system=platform.system(),
                     python=platform.python_version(), numpy=np.__version__, numpy_blas=numpy_blas()),
        cpp_build=a.label or "default")
    name = f"solver_agreement_{a.label}.json" if a.label else "solver_agreement.json"
    (problem.RESULTS / name).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
