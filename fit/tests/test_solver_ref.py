"""Tests T1 to T5 of SPEC.md section 6 for the numpy reference solver, and the pose-only case K = 0.

The helpers jacobian_error, recovery, prior_run and zero_weight_difference each measure one seed, so the
same checks can be repeated over many seeds outside pytest.
"""
import numpy as np

import solver_ref
import synth

H = 1e-6  # central-difference step of T2


def _max_abs(a):
    """Largest absolute entry; 0 for an empty array."""
    return float(np.max(np.abs(a), initial=0.0))


def _moved(R, t, theta, j, h):
    """The state moved by h along parameter j, in the column order of J: [d_omega, d_t, d_theta]."""
    if j < 3:
        e = np.zeros(3)
        e[j] = h
        return solver_ref.exp_so3(e) @ R, t, theta
    if j < 6:
        e = np.zeros(3)
        e[j - 3] = h
        return R, t + e, theta
    e = np.zeros(theta.size)
    e[j - 6] = h
    return R, t, theta + e


def jacobian_error(seed=0):
    """T2 for one seed: max abs (J - central differences) / max abs J at a perturbed state."""
    prob = synth.make_problem(seed)
    M, K = prob["P0"].shape[0], prob["theta_true"].size
    rng = np.random.default_rng(1000 + seed)
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    R = solver_ref.exp_so3(0.05 * axis) @ prob["R_true"]
    t = prob["t_true"] + 0.01 * rng.standard_normal(3)
    theta = prob["theta_true"] + 0.1 * rng.standard_normal(K)
    prob["w"] = rng.uniform(0.5, 1.5, M)
    prob["p"] = rng.uniform(0.1, 1.0, K)
    args = synth.problem_args(prob)

    _, J = solver_ref.residuals_and_jacobian(*args, R, t, theta)
    J_fd = np.empty_like(J)
    for j in range(6 + K):
        plus = solver_ref.residuals(*args, *_moved(R, t, theta, j, H))
        minus = solver_ref.residuals(*args, *_moved(R, t, theta, j, -H))
        J_fd[:, j] = (plus - minus) / (2 * H)
    return _max_abs(J - J_fd) / _max_abs(J)


def recovery(seed=0, M=120, K=12):
    """T3 for one seed: solve from init_state with p = 0; returns the result and the max abs error."""
    prob = synth.make_problem(seed, M, K)
    res = solver_ref.solve(*synth.problem_args(prob), *synth.init_state(K))
    err = max(_max_abs(res["R"] - prob["R_true"]), _max_abs(res["t"] - prob["t_true"]),
              _max_abs(res["theta"] - prob["theta_true"]))
    return res, err


def prior_run(seed=0, prior=0.5):
    """T4 for one seed: solve with p = prior; returns the result, the cost history and the gradient size
    max abs (J.T @ r) divided by norm(J) * norm(r)."""
    prob = synth.make_problem(seed)
    K = prob["theta_true"].size
    prob["p"] = np.full(K, prior)
    args = synth.problem_args(prob)
    history = []
    res = solver_ref.solve(*args, *synth.init_state(K), history=history)
    r, J = solver_ref.residuals_and_jacobian(*args, res["R"], res["t"], res["theta"])
    return res, history, _max_abs(J.T @ r) / (np.linalg.norm(J) * np.linalg.norm(r))


def zero_weight_difference(seed=0, n_out=30):
    """T5 for one seed: give n_out landmarks w = 0 and garbage x, and compare with the solve without them.

    Returns the max abs difference over R, t, theta and cost, and the two results.
    """
    prob = synth.make_problem(seed)
    P0, B, x, w, p, f, cx, cy = synth.problem_args(prob)
    M, K = P0.shape[0], p.size
    rng = np.random.default_rng(2000 + seed)
    out = rng.choice(M, size=n_out, replace=False)
    keep = np.setdiff1d(np.arange(M), out)
    w_out = w.copy()
    w_out[out] = 0.0
    x_out = x.copy()
    x_out[out] = rng.uniform(-1e4, 1e4, size=(n_out, 2))
    start = synth.init_state(K)

    full = solver_ref.solve(P0, B, x_out, w_out, p, f, cx, cy, *start)
    cut = solver_ref.solve(P0[keep], B[keep], x[keep], w[keep], p, f, cx, cy, *start)
    diff = max(_max_abs(full[key] - cut[key]) for key in ("R", "t", "theta", "cost"))
    return diff, full, cut


def test_T1_exp_so3():
    rng = np.random.default_rng(1)
    random_axes = rng.standard_normal((5, 3))
    axes = np.vstack([np.eye(3), random_axes / np.linalg.norm(random_axes, axis=1, keepdims=True)])
    for axis in axes:
        for angle in (0.0, 1e-13, 1e-6, 1.0, 3.0):
            R = solver_ref.exp_so3(angle * axis)
            assert _max_abs(R.T @ R - np.eye(3)) <= 1e-14
            assert abs(np.linalg.det(R) - 1.0) <= 1e-14
        # Just below |omega| = 1e-12 exp_so3 returns I + W, just above it uses the full formula.
        below = solver_ref.exp_so3(1e-12 * (1.0 - 1e-9) * axis)
        above = solver_ref.exp_so3(1e-12 * (1.0 + 1e-9) * axis)
        assert _max_abs(below - above) <= 1e-12


def test_T2_jacobian_fd():
    assert jacobian_error(0) <= 1e-6


def test_T3_recovery():
    res, err = recovery(0)
    assert res["status"] == "converged"
    assert err <= 1e-8


def test_T4_prior():
    # Several seeds: at the minimum the last accepted step depends on rounding, and every ending must
    # still be reported as converged.
    for seed in range(10):
        res, history, grad = prior_run(seed)
        assert res["status"] == "converged", seed
        assert len(history) == res["iterations"] + 1 and history[-1] == res["cost"]
        assert np.all(np.diff(history) <= 0.0)
        assert grad <= 1e-6, (seed, grad)


def test_T5_zero_weights():
    diff, _, _ = zero_weight_difference(0)
    assert diff <= 1e-10


def test_K0_pose_only():
    prob = synth.make_problem(seed=0, M=120, K=0)
    r, J = solver_ref.residuals_and_jacobian(*synth.problem_args(prob), *synth.init_state(0))
    assert r.shape == (240,) and J.shape == (240, 6)
    res, err = recovery(seed=0, M=120, K=0)
    assert res["status"] == "converged" and res["theta"].shape == (0,)
    assert err <= 1e-8
