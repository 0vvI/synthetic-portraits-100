"""Numpy reference solver for the head-shape fit.

This is the reference implementation that the C++ core in fitcore/ is tested against. It follows SPEC.md
sections 3 to 5.1 literally: the projection, the residuals, the analytic Jacobian with the rotation perturbed
on the left, and the Levenberg-Marquardt loop with the same damping, acceptance and stopping rules. Exactness
and a literal reading of the SPEC come before speed.

Arrays are float64 in camera axes: P0 (M, 3), B (M, 3, K), x (M, 2), w (M,), p (K,). The state is a rotation
R (3, 3), a translation t (3,) and coefficients theta (K,). K may be 0, which gives a pose-only fit.
"""
import numpy as np


def skew(a):
    """Cross-product matrix, skew(a) @ b == cross(a, b); a (..., 3) gives (..., 3, 3)."""
    a = np.asarray(a, dtype=np.float64)
    S = np.zeros(a.shape[:-1] + (3, 3))
    S[..., 0, 1] = -a[..., 2]
    S[..., 0, 2] = a[..., 1]
    S[..., 1, 0] = a[..., 2]
    S[..., 1, 2] = -a[..., 0]
    S[..., 2, 0] = -a[..., 1]
    S[..., 2, 1] = a[..., 0]
    return S


def exp_so3(omega):
    """Rotation matrix (3, 3) for the rotation vector omega (3,), by the formula of SPEC.md section 3."""
    omega = np.asarray(omega, dtype=np.float64)
    a = float(np.linalg.norm(omega))
    W = skew(omega)
    if a < 1e-12:
        return np.eye(3) + W
    return np.eye(3) + (np.sin(a) / a) * W + ((1.0 - np.cos(a)) / (a * a)) * (W @ W)


def _f64(*arrays):
    """The arguments as float64 arrays (no copy when they already are)."""
    return tuple(np.asarray(a, dtype=np.float64) for a in arrays)


def _camera_points(P0, B, R, t, theta):
    """Landmarks in camera space, Q (M, 3) with Q_i = R @ (P0[i] + B[i] @ theta) + t."""
    return (P0 + B @ theta) @ R.T + t


def _residual_vector(Q, x, w, p, f, cx, cy, theta):
    """Residuals (2M + K,): w[i] * (u_i - x[i, 0]) and w[i] * (v_i - x[i, 1]) interleaved, then p * theta."""
    M = Q.shape[0]
    u = f * Q[:, 0] / Q[:, 2] + cx
    v = f * Q[:, 1] / Q[:, 2] + cy
    r = np.empty(2 * M + theta.size)
    r[0:2 * M:2] = w * (u - x[:, 0])
    r[1:2 * M:2] = w * (v - x[:, 1])
    r[2 * M:] = p * theta
    return r


def residuals(P0, B, x, w, p, f, cx, cy, R, t, theta):
    """Residual vector r (2M + K,) at the state (R, t, theta)."""
    P0, B, x, w, p, R, t, theta = _f64(P0, B, x, w, p, R, t, theta)
    Q = _camera_points(P0, B, R, t, theta)
    return _residual_vector(Q, x, w, p, float(f), float(cx), float(cy), theta)


def residuals_and_jacobian(P0, B, x, w, p, f, cx, cy, R, t, theta):
    """Residuals r (2M + K,) and Jacobian J (2M + K, 6 + K) with columns [d_omega, d_t, d_theta].

    The rotation columns are the derivative of r at d_omega = 0 for the left perturbation
    R <- exp_so3(d_omega) @ R. The r returned here is bit-identical to residuals() at the same state.
    """
    P0, B, x, w, p, R, t, theta = _f64(P0, B, x, w, p, R, t, theta)
    f = float(f)
    Q = _camera_points(P0, B, R, t, theta)
    r = _residual_vector(Q, x, w, p, f, float(cx), float(cy), theta)
    M, K = Q.shape[0], theta.size

    s = f / Q[:, 2]
    D = np.zeros((M, 2, 3))
    D[:, 0, 0] = s
    D[:, 1, 1] = s
    D[:, 0, 2] = s * (-Q[:, 0] / Q[:, 2])
    D[:, 1, 2] = s * (-Q[:, 1] / Q[:, 2])

    # D_i @ [-skew(Q_i - t) | I_3 | R @ B[i]] for all landmarks at once; D_i @ I_3 is D_i itself.
    rows = np.concatenate([D @ (-skew(Q - t)), D, D @ (R @ B)], axis=2)
    J = np.zeros((2 * M + K, 6 + K))
    J[:2 * M] = (w[:, None, None] * rows).reshape(2 * M, 6 + K)
    J[2 * M:, 6:] = np.diag(p)
    return r, J


def solve(P0, B, x, w, p, f, cx, cy, R, t, theta,
          max_iter=100, lam0=1e-3, ftol=1e-14, xtol=1e-12, *, history=None):
    """Levenberg-Marquardt fit of (R, t, theta) by the algorithm of SPEC.md section 4.

    Returns a dict with "R" (3, 3), "t" (3,), "theta" (K,), "cost" (float), "iterations" (the number of
    accepted steps) and "status" ("converged", "max_iter" or "stalled"). If history is a list, the initial
    cost and then the cost after every accepted step are appended to it. The inputs are not modified.
    """
    P0, B, x, w, p = _f64(P0, B, x, w, p)
    problem = (P0, B, x, w, p, float(f), float(cx), float(cy))
    R, t, theta = (np.array(a, dtype=np.float64, order="C") for a in (R, t, theta))

    lam = float(lam0)
    r, J = residuals_and_jacobian(*problem, R, t, theta)
    cost = 0.5 * float(r @ r)
    if history is not None:
        history.append(cost)
    status = "max_iter"
    iterations = 0
    for _ in range(max_iter):
        A = J.T @ J
        g = J.T @ r
        accepted = False
        while True:
            Ad = A + lam * np.diag(np.maximum(np.diag(A), 1e-12))
            delta = np.linalg.solve(Ad, -g)
            # A candidate can put a landmark at Q.z = 0; its cost is then not finite and it is rejected
            # below, so the floating-point warnings it would raise are silenced.
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                cand = (exp_so3(delta[0:3]) @ R, t + delta[3:6], theta + delta[6:])
                r_c = residuals(*problem, *cand)
                cost_c = 0.5 * float(r_c @ r_c)
            if np.isfinite(cost_c) and cost_c < cost:
                rel = (cost - cost_c) / max(cost, 1e-300)
                (R, t, theta), cost = cand, cost_c
                lam = max(lam / 10, 1e-15)
                accepted = True
                break
            if np.max(np.abs(delta)) < xtol:
                # A rejected step this small means the cost cannot be lowered any further in floating
                # point: the state is at the minimum, not stuck.
                status = "converged"
                break
            lam = lam * 10
            if lam > 1e12:
                break
        if status == "converged":
            break
        if not accepted:
            status = "stalled"
            break
        iterations += 1
        if history is not None:
            history.append(cost)
        r, J = residuals_and_jacobian(*problem, R, t, theta)
        if rel < ftol or np.max(np.abs(delta)) < xtol:
            status = "converged"
            break
    return {"R": R, "t": t, "theta": theta, "cost": cost, "iterations": iterations, "status": status}
