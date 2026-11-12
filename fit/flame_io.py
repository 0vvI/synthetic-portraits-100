"""Turn the FLAME 2023 Open download into the small array files the fit uses.

FLAME 2023 Open (Li et al., Max Planck Institute for Intelligent Systems) is CC BY 4.0. Its files are not
redistributed in this repository: download the model from https://flame.is.tue.mpg.de and run

    python fit/flame_io.py --pkl /path/to/flame2023_Open.pkl [--masks /path/to/FLAME_masks.pkl]

This writes fit/assets/flame_min.npz (and flame_masks.npz), which git ignores.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parent / "assets"
N_SHAPE, N_EXPR = 300, 100
R_FLIP = np.diag([1.0, -1.0, -1.0])  # FLAME axes (x left, y up, z out of the face) -> camera axes


def prepare(pkl, masks=None, out_dir=ASSETS):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(pkl, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    sd = np.asarray(d["shapedirs"], dtype=np.float64)
    assert sd.shape == (5023, 3, N_SHAPE + N_EXPR), sd.shape
    np.savez(
        out_dir / "flame_min.npz",
        v_template=np.asarray(d["v_template"], dtype=np.float64),
        f=np.asarray(d["f"], dtype=np.int64),
        shapedirs_shape=sd[:, :, :N_SHAPE],
        shapedirs_expr=sd[:, :, N_SHAPE:],
        source="FLAME 2023 Open (flame2023_Open.pkl); metres; y up; face looks along +z",
    )
    if masks:
        with open(masks, "rb") as fh:
            m = pickle.load(fh, encoding="latin1")
        np.savez(out_dir / "flame_masks.npz", **{k: np.asarray(v, dtype=np.int64) for k, v in m.items()})


def load_flame(path=ASSETS / "flame_min.npz"):
    """Return (V0 (5023, 3), F (9976, 3), S (5023, 3, 300), E (5023, 3, 100)) in FLAME axes, metres."""
    d = np.load(path)
    return d["v_template"], d["f"], d["shapedirs_shape"], d["shapedirs_expr"]


def landmark_model(embedding_npz, n_shape, n_expr, flame_npz=ASSETS / "flame_min.npz"):
    """Landmark-level linear model in camera axes: P0 (M, 3) and B (M, 3, n_shape + n_expr). See SPEC.md 7."""
    V0, F, S, E = load_flame(flame_npz)
    emb = np.load(embedding_npz)
    tri = F[emb["face_idx"]]                      # (M, 3) vertex ids
    bary = emb["bary"]                            # (M, 3)
    P0 = np.einsum("mj,mjc->mc", bary, V0[tri])
    basis = np.concatenate([S[:, :, :n_shape], E[:, :, :n_expr]], axis=2)
    B = np.einsum("mj,mjck->mck", bary, basis[tri])
    P0 = np.ascontiguousarray(P0 @ R_FLIP.T)
    B = np.ascontiguousarray(np.einsum("dc,mck->mdk", R_FLIP, B))
    return P0, B


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--masks")
    ap.add_argument("--out-dir", default=str(ASSETS))
    a = ap.parse_args()
    prepare(a.pkl, a.masks, a.out_dir)
    print("wrote", Path(a.out_dir) / "flame_min.npz")
