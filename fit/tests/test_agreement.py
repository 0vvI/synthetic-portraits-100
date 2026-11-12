"""T6 and T7 of SPEC.md: the C++ core against the numpy reference, and solve_batch against a loop.

Skipped when the fitcore Python module has not been built. Build it with -DFITCORE_PYTHON=ON and, if the
build directory is not build/fitcore under the repository root, point FITCORE_BUILD_DIR at it.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import solver_ref
import synth

BUILD = Path(os.environ.get("FITCORE_BUILD_DIR", Path(__file__).resolve().parents[2] / "build" / "fitcore"))
sys.path.insert(0, str(BUILD.resolve()))
fitcore = pytest.importorskip("fitcore", reason="fitcore Python module not built")
if not hasattr(fitcore, "solve_batch"):
    # Without the built module, "import fitcore" finds the source folder fit/fitcore as an empty package.
    pytest.skip("fitcore Python module not built", allow_module_level=True)


def rel_diff(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300)


def perturbed_state(prob, rng):
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    R = solver_ref.exp_so3(0.05 * axis) @ prob["R_true"]
    t = prob["t_true"] + 0.01 * rng.standard_normal(3)
    theta = prob["theta_true"] + 0.1 * rng.standard_normal(prob["theta_true"].shape)
    return R, t, theta


@pytest.mark.parametrize("M,K", [(120, 12), (468, 110), (120, 0)])
def test_T6_residuals_and_jacobian_agree(M, K):
    rng = np.random.default_rng(7)
    prob = synth.make_problem(seed=3, M=M, K=K)
    prob["w"] = rng.uniform(0.5, 1.5, M)
    prob["p"] = rng.uniform(0.1, 1.0, K)
    args = synth.problem_args(prob)
    state = perturbed_state(prob, rng)

    r_ref, J_ref = solver_ref.residuals_and_jacobian(*args, *state)
    r_cpp, J_cpp = fitcore.residuals_and_jacobian(*args, *state)
    assert r_cpp.shape == r_ref.shape and J_cpp.shape == J_ref.shape
    assert rel_diff(r_ref, r_cpp) <= 1e-9
    assert rel_diff(J_ref, J_cpp) <= 1e-9
    assert rel_diff(r_ref, fitcore.residuals(*args, *state)) <= 1e-9


@pytest.mark.parametrize("M,K,prior", [(120, 12, 0.0), (120, 12, 0.5), (468, 110, 0.05), (120, 0, 0.0)])
def test_T6_solutions_agree(M, K, prior):
    prob = synth.make_problem(seed=1, M=M, K=K)
    prob["p"] = np.full(K, prior)
    args = synth.problem_args(prob)
    init = synth.init_state(K)

    ref = solver_ref.solve(*args, *init)
    cpp = fitcore.solve(*args, *init)
    assert ref["status"] == cpp["status"] == "converged"
    assert np.max(np.abs(ref["R"] - cpp["R"])) <= 1e-8
    assert np.max(np.abs(ref["t"] - cpp["t"])) <= 1e-8
    if K:
        assert np.max(np.abs(ref["theta"] - cpp["theta"])) <= 1e-8
    # Relative for a problem with a residual; the exact-data cases end at rounding noise (cost near 1e-20),
    # where only an absolute bound means anything.
    assert abs(ref["cost"] - cpp["cost"]) <= 1e-10 * ref["cost"] + 1e-12


@pytest.mark.parametrize("n_threads", [1, 4, 0])
def test_T7_batch_equals_loop(n_threads):
    N, M, K = 8, 120, 12
    base = synth.make_problem(seed=0, M=M, K=K)
    base["p"] = np.full(K, 0.2)
    P0, B, _, _, p, f, cx, cy = synth.problem_args(base)
    rng = np.random.default_rng(11)
    xs = np.stack([base["x"] + rng.normal(0.0, 0.5, base["x"].shape) for _ in range(N)])
    ws = rng.uniform(0.5, 1.5, (N, M))
    R0, t0, th0 = synth.init_state(K)
    Rs, ts, thetas = np.stack([R0] * N), np.stack([t0] * N), np.stack([th0] * N)

    batch = fitcore.solve_batch(P0, B, xs, ws, p, f, cx, cy, Rs, ts, thetas, n_threads=n_threads)
    assert len(batch) == N
    for i in range(N):
        one = fitcore.solve(P0, B, xs[i], ws[i], p, f, cx, cy, R0, t0, th0)
        for key in ("R", "t", "theta"):
            assert np.array_equal(one[key], batch[i][key]), (i, key)
        assert one["cost"] == batch[i]["cost"] and one["iterations"] == batch[i]["iterations"]
        assert one["status"] == batch[i]["status"]
