"""Shared pieces of the head-shape fit: the data, the landmark split, the error measure and the solver choice.

Everything here follows fit/SPEC.md sections 7 and 8. The scripts that use it are run_fit.py (the stages and
the evaluation), compare_solvers.py (C++ against numpy on the real problems) and figures.py.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import flame_io
import solver_ref

FIT = Path(__file__).resolve().parent
ROOT = FIT.parent
RESULTS = FIT / "results"
EMBEDDING = FIT / "embedding" / "mediapipe468_on_flame2023.npz"

SPLIT_SEED = 20261111
N_LANDMARKS = 468
DIAGONAL_35MM = float(np.hypot(36.0, 24.0))   # mm; focal lengths are quoted as 35 mm equivalents
SOLVE_OPTIONS = dict(max_iter=200, lam0=1e-3, ftol=1e-12, xtol=1e-12)
REGIONS = ("face_oval", "lips", "nose", "eyes", "brows")


def focal_px(focal_mm, width, height):
    """Focal length in pixels for a 35 mm-equivalent focal length, matched on the image diagonal."""
    return focal_mm * float(np.hypot(width, height)) / DIAGONAL_35MM


def load_landmarks():
    """ids (N,), xy (N, 468, 2) float64 pixels at full size, and the image size (W, H)."""
    d = np.load(RESULTS / "landmarks_mp478.npz")
    sizes = np.unique(d["size"], axis=0)
    assert len(sizes) == 1, "the portraits are expected to share one size"
    return d["ids"], d["xy"][:, :N_LANDMARKS].astype(np.float64), (int(sizes[0, 0]), int(sizes[0, 1]))


def load_regions():
    """Landmark ids per face region (written by regions.py from MediaPipe's own tables)."""
    return json.loads((RESULTS / "regions_mp468.json").read_text())


def load_valid():
    """Which of the 468 landmarks have a usable position on the FLAME surface."""
    return np.load(EMBEDDING)["valid"].astype(bool)


def split_masks():
    """The one split of the 468 landmark ids: boolean masks "fit" (60 %), "val" (20 %), "test" (20 %)."""
    order = np.random.default_rng(SPLIT_SEED).permutation(N_LANDMARKS)
    n_fit, n_val = round(0.6 * N_LANDMARKS), round(0.2 * N_LANDMARKS)
    parts = {"fit": order[:n_fit], "val": order[n_fit:n_fit + n_val], "test": order[n_fit + n_val:]}
    masks = {}
    for name, ids in parts.items():
        masks[name] = np.zeros(N_LANDMARKS, bool)
        masks[name][ids] = True
    return masks


def interocular(xy, regions):
    """Distance in pixels between the means of the right-eye and left-eye contour landmarks, per portrait."""
    right, left = xy[:, regions["right_eye"]].mean(1), xy[:, regions["left_eye"]].mean(1)
    return np.linalg.norm(right - left, axis=1)


def landmark_model(n_shape, n_expr):
    """P0 (468, 3) and B (468, 3, n_shape + n_expr) in camera axes; needs fit/assets/flame_min.npz."""
    return flame_io.landmark_model(EMBEDDING, n_shape, n_expr)


def initial_pose(P0, x, use, f, cx, cy):
    """A head facing the camera, at the depth where the model's landmarks spread as wide as the observed ones.

    P0 is already in camera axes, so the rotation starts at the identity.
    """
    model, seen = P0[use], x[use]
    centre_m, centre_x = model.mean(0), seen.mean(0)
    spread_m = np.sqrt(((model[:, :2] - centre_m[:2]) ** 2).sum(1).mean())
    spread_x = np.sqrt(((seen - centre_x) ** 2).sum(1).mean())
    z = f * spread_m / spread_x
    t = np.array([(centre_x[0] - cx) * z / f - centre_m[0], (centre_x[1] - cy) * z / f - centre_m[1],
                  z - centre_m[2]])
    return np.eye(3), t


def project(P0, B, f, cx, cy, R, t, theta):
    """Pixel positions (M, 2) of the model's landmarks at the state (R, t, theta)."""
    Q = (P0 + B @ theta) @ R.T + t
    return np.stack([f * Q[:, 0] / Q[:, 2] + cx, f * Q[:, 1] / Q[:, 2] + cy], axis=1)


def landmark_error(pred, seen, iod, mask):
    """Mean distance between predicted and observed landmarks over mask, in percent of that portrait's
    inter-ocular distance."""
    return float(np.linalg.norm(pred[mask] - seen[mask], axis=1).mean() / iod * 100.0)


def load_fitcore():
    """The C++ module, or None when it has not been built (see fit/fitcore/README.md)."""
    build = Path(os.environ.get("FITCORE_BUILD_DIR", ROOT / "build" / "fitcore")).resolve()
    if str(build) not in sys.path:
        sys.path.insert(0, str(build))
    try:
        import fitcore
    except ImportError:
        return None
    return fitcore if hasattr(fitcore, "solve_batch") else None


class Solver:
    """One interface over the two implementations: name "cpp" (fitcore), "ref" (numpy) or "auto"."""

    def __init__(self, name="auto"):
        core = load_fitcore() if name in ("auto", "cpp") else None
        if name == "cpp" and core is None:
            raise SystemExit("the fitcore module is not built; build it or use --solver ref")
        self.core = core
        self.name = "cpp" if core is not None else "ref"

    def solve_many(self, P0, B, xs, ws, p, f, cx, cy, Rs, ts, thetas):
        """Fit N portraits that share one landmark model; returns a list of result dicts."""
        if self.core is not None:
            return self.core.solve_batch(P0, B, xs, ws, p, f, cx, cy, Rs, ts, thetas, **SOLVE_OPTIONS)
        return [solver_ref.solve(P0, B, xs[i], ws[i], p, f, cx, cy, Rs[i], ts[i], thetas[i], **SOLVE_OPTIONS)
                for i in range(len(xs))]
