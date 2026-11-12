"""Synthetic problem of SPEC.md section 6, shared by the tests of the numpy reference and of the C++ core.

make_problem draws a random landmark model, a true pose and true coefficients, and returns the exact
(noise-free) projections as the observations. It uses numpy only and does not call either solver.
"""
import numpy as np

F, CX, CY = 1500.0, 320.0, 240.0


def _axis_angle(axis, angle):
    """Rotation by angle (radians) about the unit vector axis, by the Rodrigues formula."""
    k = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def make_problem(seed=0, M=120, K=12):
    """Problem dict with keys P0, B, x, w, p, f, cx, cy, R_true, t_true, theta_true (w = 1, p = 0).

    P0 is uniform in a 0.2 m cube centred on the origin, B is standard normal times 0.01, the true rotation
    is 0.3 rad about a random axis, t_true = (0.02, -0.01, 0.6) and theta_true is standard normal. K may be 0.
    """
    rng = np.random.default_rng(seed)
    P0 = rng.uniform(-0.1, 0.1, size=(M, 3))
    B = 0.01 * rng.standard_normal((M, 3, K))
    axis = rng.standard_normal(3)
    axis /= np.linalg.norm(axis)
    R_true = _axis_angle(axis, 0.3)
    t_true = np.array([0.02, -0.01, 0.6])
    theta_true = rng.standard_normal(K)

    Q = (P0 + B @ theta_true) @ R_true.T + t_true
    x = np.empty((M, 2))
    x[:, 0] = F * Q[:, 0] / Q[:, 2] + CX
    x[:, 1] = F * Q[:, 1] / Q[:, 2] + CY
    return {"P0": P0, "B": B, "x": x, "w": np.ones(M), "p": np.zeros(K), "f": F, "cx": CX, "cy": CY,
            "R_true": R_true, "t_true": t_true, "theta_true": theta_true}


def init_state(K):
    """The T3 start (R, t, theta): identity rotation, t = (0, 0, 0.5), theta = 0."""
    return np.eye(3), np.array([0.0, 0.0, 0.5]), np.zeros(K)


def problem_args(prob):
    """(P0, B, x, w, p, f, cx, cy) in the argument order of SPEC.md 5.1, for calling either solver."""
    return tuple(prob[key] for key in ("P0", "B", "x", "w", "p", "f", "cx", "cy"))
