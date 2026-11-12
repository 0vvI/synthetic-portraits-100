"""Build the landmark embedding: where each of MediaPipe's 468 face-mesh landmarks sits on FLAME 2023 Open.

    python fit/embedding/build_embedding.py [--flame F] [--masks M] [--canonical OBJ] [--out NPZ]

Writes mediapipe468_on_flame2023.npz and a .csv with the same stem (format in fit/SPEC.md section 7): for each
vertex of MediaPipe's canonical face mesh (vertex n is landmark n) one FLAME triangle, barycentric weights, a
validity flag and the distance left after the alignment, in millimetres of FLAME's own scale.

Steps:
1. Read the canonical mesh (centimetres -> metres) and confirm that its axes agree with FLAME's.
2. Allowed FLAME triangles: every triangle without an eyeball vertex.
3. Rough similarity: scale from the distance between the eye centres, no rotation, translation so that the
   eye midpoints agree in x and y and the nose tips agree in z.
4. Closest-point loop with a normal test. First the similarity alone until it settles; then rounds that fit
   100 shape and 20 expression coefficients with a ridge prior (similarity re-estimated in each round) and
   recompute the closest points on the deformed FLAME surface. The canonical mesh stays fixed; FLAME moves.
   The ridge weight is the smallest value on a fixed ladder for which no coefficient reaches 3.
5. Closest normal-compatible point on the fitted surface -> triangle and barycentric weights.

Checks E1 to E4 are printed at the end. Deterministic; numpy and scipy only. No FLAME vertex positions or basis
values are written, only triangle indices, barycentric weights, flags and distances.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
FIT = HERE.parent
DEFAULT_FLAME = FIT / "assets" / "flame_min.npz"
DEFAULT_MASKS = FIT / "assets" / "flame_masks.npz"
DEFAULT_CANONICAL = HERE / "third_party" / "canonical_face_model.obj"
DEFAULT_OUT = HERE / "mediapipe468_on_flame2023.npz"

# MediaPipe landmark ids (the subject's right is the image left).
NOSE_TIP, CHIN, FOREHEAD_TOP = 1, 152, 10
MOUTH_RIGHT, MOUTH_LEFT = 61, 291
RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_RING = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
INNER_LIP_UPPER = [191, 80, 81, 82, 13, 312, 311, 310, 415]
INNER_LIP_LOWER = [95, 88, 178, 87, 14, 317, 402, 318, 324]

N_SHAPE, N_EXPR = 100, 20          # coefficients fitted in step 4
NORMAL_MIN_DOT = 0.3               # normal test: triangle normal . canonical vertex normal >= this
CANDIDATE_MARGIN = 0.006           # metres, FLAME scale: candidates lie within this of the plain closest distance
RESIDUAL_MAX_MM = 6.0              # larger residual -> landmark invalid
TOL_MM = 0.01                      # stop when the mean distance changes by less than this between rounds
MAX_ROUNDS = 30                    # rounds of (coefficients + similarity, new closest points)
MAX_SIMILARITY_ROUNDS = 100        # similarity-only rounds before that
INNER_STEPS = 5                    # coefficient / similarity alternations inside one round
RIDGE_LADDER = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)  # mm^2
COEF_LIMIT = 3.0
EYE_FILL_RADIUS = 0.016            # metres: unmasked vertices this close to an eyeball centre fill the eye opening


def load_obj(path):
    """Vertices (n, 3) in metres and triangles (m, 3) of a Wavefront OBJ given in centimetres."""
    verts, faces = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(t) for t in parts[1:4]])
            elif parts[0] == "f":
                faces.append([int(t.split("/")[0]) - 1 for t in parts[1:]])
    return np.array(verts) / 100.0, np.array(faces, dtype=np.int64)


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def vertex_normals(V, F):
    """Area-weighted vertex normals (the cross product of two edges is twice the area times the normal)."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    acc = np.zeros_like(V)
    for j in range(3):
        np.add.at(acc, F[:, j], n)
    return unit(acc)


def closest_on_triangles(p, a, b, c):
    """Closest point on triangle (a, b, c) to p, for n pairs given as (n, 3) arrays.

    Returns barycentric weights (n, 3) and squared distances (n,). The region tests follow Ericson,
    Real-Time Collision Detection, section 5.1.5; the first region that applies wins.
    """
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    d3, d4 = np.einsum("ij,ij->i", ab, bp), np.einsum("ij,ij->i", ac, bp)
    d5, d6 = np.einsum("ij,ij->i", ab, cp), np.einsum("ij,ij->i", ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    bary = np.zeros((len(p), 3))
    done = (d1 <= 0) & (d2 <= 0)                                       # vertex a
    bary[done, 0] = 1.0
    m = ~done & (d3 >= 0) & (d4 <= d3)                                 # vertex b
    bary[m, 1] = 1.0
    done |= m
    m = ~done & (vc <= 0) & (d1 >= 0) & (d3 <= 0)                      # edge ab
    v = d1[m] / (d1[m] - d3[m])
    bary[m, 0], bary[m, 1] = 1.0 - v, v
    done |= m
    m = ~done & (d6 >= 0) & (d5 <= d6)                                 # vertex c
    bary[m, 2] = 1.0
    done |= m
    m = ~done & (vb <= 0) & (d2 >= 0) & (d6 <= 0)                      # edge ac
    w = d2[m] / (d2[m] - d6[m])
    bary[m, 0], bary[m, 2] = 1.0 - w, w
    done |= m
    m = ~done & (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)            # edge bc
    w = (d4 - d3)[m] / ((d4 - d3)[m] + (d5 - d6)[m])
    bary[m, 1], bary[m, 2] = 1.0 - w, w
    done |= m
    m = ~done                                                          # inside
    den = va[m] + vb[m] + vc[m]
    v, w = vb[m] / den, vc[m] / den
    bary[m, 0], bary[m, 1], bary[m, 2] = 1.0 - v - w, v, w
    q = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:] * c
    return bary, ((p - q) ** 2).sum(1)


class Surface:
    """The allowed FLAME triangles at one deformation, with a k-d tree over the triangle centroids."""

    def __init__(self, V, F, allowed):
        self.ids = allowed                                   # indices into F
        self.T = V[F[allowed]]                               # (n, 3 corners, 3)
        self.g = self.T.mean(1)
        self.rho = np.linalg.norm(self.T - self.g[:, None], axis=2).max(1)   # centroid-to-corner radius
        self.normal = unit(np.cross(self.T[:, 1] - self.T[:, 0], self.T[:, 2] - self.T[:, 0]))
        self.tree = cKDTree(self.g)

    def query(self, P, N):
        """Closest normal-compatible point for each row of P (FLAME frame) whose unit normal is N.

        Candidates are all allowed triangles that come within CANDIDATE_MARGIN of the plain closest distance;
        they are collected exactly with the bound |p - centroid| - radius <= distance. A point fails the
        normal test when no candidate passes it; it then takes the plain closest point.
        Returns (triangle ids into F, barycentric weights, distances, failed flags, plain distances).
        """
        k = 8
        _, nn = self.tree.query(P, k=k)
        pi, ti = np.repeat(np.arange(len(P)), k), nn.ravel()
        _, d2 = closest_on_triangles(P[pi], self.T[ti, 0], self.T[ti, 1], self.T[ti, 2])
        reach = np.sqrt(d2.reshape(-1, k).min(1)) + CANDIDATE_MARGIN   # >= plain distance + margin
        lists = self.tree.query_ball_point(P, reach + self.rho.max(), return_sorted=True)
        pi = np.concatenate([np.full(len(l), i, dtype=np.int64) for i, l in enumerate(lists)])
        ti = np.concatenate([np.asarray(l, dtype=np.int64) for l in lists])
        keep = np.linalg.norm(P[pi] - self.g[ti], axis=1) - self.rho[ti] <= reach[pi]
        pi, ti = pi[keep], ti[keep]
        bary, d2 = closest_on_triangles(P[pi], self.T[ti, 0], self.T[ti, 1], self.T[ti, 2])
        d = np.sqrt(d2)
        first = lambda key: np.lexsort((key, pi))[np.r_[0, np.flatnonzero(np.diff(pi)) + 1]]
        plain = first(d)                                         # pi is sorted, so groups keep their order
        ok = (np.einsum("ij,ij->i", self.normal[ti], N[pi]) >= NORMAL_MIN_DOT) & (d <= d[plain][pi] + CANDIDATE_MARGIN)
        best = first(np.where(ok, d, np.inf))
        failed = ~ok[best]
        pick = np.where(failed, plain, best)
        return self.ids[ti[pick]], bary[pick], d[pick], failed, d[plain]


def umeyama(X, Y):
    """Similarity (s, R, t) minimising sum |s R x + t - y|^2 (Umeyama 1991)."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    U, D, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    E = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ E @ Vt
    s = np.trace(np.diag(D) @ E) / (Xc ** 2).sum(1).mean()
    return s, R, Y.mean(0) - s * R @ mx


def to_flame(C, Nc, s, R, t):
    """Canonical points and normals into the FLAME frame, given the FLAME -> canonical similarity."""
    return ((C - t) @ R) / s, Nc @ R


def check_axes(C, F):
    """Step 1: flip any canonical axis that disagrees with FLAME's; return (C, F, notes)."""
    notes, flips = [], 0
    if C[FOREHEAD_TOP, 1] < C[CHIN, 1]:
        C[:, 1] *= -1
        flips += 1
        notes.append("flipped y")
    if C[NOSE_TIP, 2] < np.median(C[:, 2]):
        C[:, 2] *= -1
        flips += 1
        notes.append("flipped z")
    if C[LEFT_EYE_RING[0], 0] < C[RIGHT_EYE_RING[0], 0]:
        C[:, 0] *= -1
        flips += 1
        notes.append("flipped x")
    if flips % 2:
        F = F[:, ::-1].copy()
    top = np.argsort(-C[:, 2])[:2]
    notes.append(f"y: forehead 10 at {C[FOREHEAD_TOP, 1] * 100:.2f} cm above chin 152 at {C[CHIN, 1] * 100:.2f} cm")
    notes.append(f"z: largest z at landmark {top[0]} ({C[top[0], 2] * 100:.3f} cm), next landmark {top[1]} "
                 f"({C[top[1], 2] * 100:.3f} cm)")
    notes.append(f"x: 263 at {C[263, 0] * 100:.2f} cm, 33 at {C[33, 0] * 100:.2f} cm")
    if not notes[0].startswith("flipped"):
        notes.insert(0, "axes agree with FLAME's, no flip")
    return C, F, notes


def flame_eye_centres(V0, F, masks):
    """Mean of the eye-region vertices on the border of each eye opening.

    FLAME 2023 has no holes at the eyes: the opening behind the lids is filled by vertices that belong to no
    mask. The border of the left (right) opening is the set of left_eye_region (right_eye_region) vertices
    that share a triangle with such a fill vertex lying within EYE_FILL_RADIUS of that eyeball's centre.
    """
    in_any = np.zeros(len(V0), bool)
    for k in masks.files:
        in_any[masks[k]] = True
    out = []
    for side in ("left", "right"):
        centre = V0[masks[f"{side}_eyeball"]].mean(0)
        fill = ~in_any & (np.linalg.norm(V0 - centre, axis=1) < EYE_FILL_RADIUS)
        region = np.zeros(len(V0), bool)
        region[masks[f"{side}_eye_region"]] = True
        touch = F[fill[F].any(1)]
        border = np.unique(touch[region[touch]])
        out.append((V0[border].mean(0), len(border), int(fill.sum())))
    return out


def fit_round_coefficients(P0, Bk, C, s, R, t, lam):
    """Inside one round: alternate the ridge solve for the coefficients and the similarity."""
    K = Bk.shape[2]
    for _ in range(INNER_STEPS):
        J = 1000.0 * s * np.einsum("dc,ick->idk", R, Bk).reshape(-1, K)     # mm per coefficient
        r0 = (1000.0 * (s * P0 @ R.T + t - C)).ravel()                       # mm
        theta = np.linalg.solve(J.T @ J + lam * np.eye(K), -J.T @ r0)
        s, R, t = umeyama(P0 + Bk @ theta, C)
    return theta, s, R, t


def run_icp(V0, F, D, allowed, C, Nc, sim, lam, log=None):
    """Step 4 (alternating part) for one ridge weight. Returns the final state and the last correspondences."""
    s, R, t = sim
    theta = np.zeros(D.shape[2])
    surf = Surface(V0, F, allowed)
    corr = surf.query(*to_flame(C, Nc, s, R, t))
    prev = corr[2].mean() * 1000
    max_seen = 0.0
    rounds = 0
    for rounds in range(1, MAX_ROUNDS + 1):
        tri = F[corr[0]]
        P0 = np.einsum("ij,ijc->ic", corr[1], V0[tri])
        Bk = np.einsum("ij,ijck->ick", corr[1], D[tri])
        theta, s, R, t = fit_round_coefficients(P0, Bk, C, s, R, t, lam)
        max_seen = max(max_seen, np.abs(theta).max())
        V = V0 + D @ theta
        surf = Surface(V, F, allowed)
        corr = surf.query(*to_flame(C, Nc, s, R, t))
        mean = corr[2].mean() * 1000
        if log is not None:
            log.append((rounds, mean, np.abs(theta).max(), s))
        if abs(prev - mean) < TOL_MM:
            prev = mean
            break
        prev = mean
    return dict(theta=theta, s=s, R=R, t=t, corr=corr, rounds=rounds, mean_mm=prev, max_seen=max_seen, V=V)


def landmark_positions(V, F, face_idx, bary):
    return np.einsum("ij,ijc->ic", bary, V[F[face_idx]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flame", default=str(DEFAULT_FLAME))
    ap.add_argument("--masks", default=str(DEFAULT_MASKS))
    ap.add_argument("--canonical", default=str(DEFAULT_CANONICAL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    a = ap.parse_args()

    # Step 1: canonical mesh.
    C, Fc = load_obj(a.canonical)
    assert C.shape == (468, 3), C.shape
    C, Fc, notes = check_axes(C, Fc)
    print("step 1:", "; ".join(notes))
    Nc = vertex_normals(C, Fc)

    # Step 2: FLAME and the allowed triangles.
    fl = np.load(a.flame)
    V0, F = fl["v_template"], fl["f"].astype(np.int64)
    D = np.concatenate([fl["shapedirs_shape"][:, :, :N_SHAPE], fl["shapedirs_expr"][:, :, :N_EXPR]], axis=2)
    masks = np.load(a.masks)
    eyeball = np.zeros(len(V0), bool)
    eyeball[masks["left_eyeball"]] = True
    eyeball[masks["right_eyeball"]] = True
    allowed = np.flatnonzero(~eyeball[F].any(1))
    print(f"step 2: {len(allowed)} of {len(F)} FLAME triangles allowed (the rest touch an eyeball)")

    # Step 3: rough similarity, FLAME -> canonical: c = s R x + t.
    eye_c = [C[LEFT_EYE_RING].mean(0), C[RIGHT_EYE_RING].mean(0)]
    (eye_fl, n_bl, n_fl), (eye_fr, n_br, n_fr) = flame_eye_centres(V0, F, masks)
    s = np.linalg.norm(eye_c[0] - eye_c[1]) / np.linalg.norm(eye_fl - eye_fr)
    R = np.eye(3)
    nose = masks["nose"]
    tip_fl = V0[nose[np.argmax(V0[nose, 2])]]
    mid_c, mid_fl = (eye_c[0] + eye_c[1]) / 2, (eye_fl + eye_fr) / 2
    t = np.array([mid_c[0] - s * mid_fl[0], mid_c[1] - s * mid_fl[1], C[:, 2].max() - s * tip_fl[2]])
    print(f"step 3: FLAME eye-opening border {n_bl} + {n_br} vertices (fill {n_fl} + {n_fr}); "
          f"eye distance canonical {np.linalg.norm(eye_c[0] - eye_c[1]) * 1000:.2f} mm, "
          f"FLAME {np.linalg.norm(eye_fl - eye_fr) * 1000:.2f} mm; initial scale {s:.4f}")

    # Step 4a: similarity only, on the template.
    surf0 = Surface(V0, F, allowed)
    prev, n_sim = None, 0
    for n_sim in range(1, MAX_SIMILARITY_ROUNDS + 1):
        fidx, bary, dist, failed, _ = surf0.query(*to_flame(C, Nc, s, R, t))
        mean = dist.mean() * 1000
        if prev is not None and abs(prev - mean) < TOL_MM:
            break
        prev = mean
        s, R, t = umeyama(landmark_positions(V0, F, fidx, bary), C)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"step 4: similarity only: {n_sim} rounds, mean distance {mean:.3f} mm, scale {s:.4f}, "
          f"rotation {ang:.2f} deg, normal-test failures {failed.sum()}")

    # Step 4b: alternate coefficients and correspondences; smallest ridge weight that keeps |coef| < 3.
    chosen = None
    for lam in RIDGE_LADDER:
        log = []
        res = run_icp(V0, F, D, allowed, C, Nc, (s, R, t), lam, log)
        top = np.abs(res["theta"]).max()
        print(f"  ridge {lam:g} mm^2: {res['rounds']} rounds, mean distance {res['mean_mm']:.3f} mm, "
              f"max |coef| final {top:.3f}, over rounds {res['max_seen']:.3f}")
        if res["max_seen"] < COEF_LIMIT:
            chosen = (lam, res, log)
            break
    if chosen is None:
        raise SystemExit("no ridge weight on the ladder keeps every coefficient below 3")
    lam, res, log = chosen
    theta = res["theta"]
    k_top = int(np.argmax(np.abs(theta)))
    name = f"shape {k_top}" if k_top < N_SHAPE else f"expression {k_top - N_SHAPE}"
    print(f"step 4: ridge weight {lam:g} mm^2 per coefficient^2; {res['rounds']} rounds; "
          f"largest |coef| {abs(theta[k_top]):.3f} ({name}); similarity scale {res['s']:.4f} "
          f"(FLAME -> canonical)")
    for r in log:
        print(f"    round {r[0]:2d}: mean distance {r[1]:.4f} mm, max |coef| {r[2]:.3f}, scale {r[3]:.4f}")

    # Step 5: the asset.
    face_idx, bary, dist, failed, _ = res["corr"]
    bary = np.clip(bary, 0.0, None)
    bary /= bary.sum(1, keepdims=True)
    residual_mm = dist * 1000.0
    face_idx = face_idx.astype(np.int32)
    reasons = {i: [] for i in range(468)}
    for i in np.flatnonzero(residual_mm > RESIDUAL_MAX_MM):
        reasons[i].append(f"residual {residual_mm[i]:.2f} mm")
    for i in np.flatnonzero(failed):
        reasons[i].append("normal test")

    # Semantic checks that decide validity (E4c, E4d), on the undeformed template.
    Pt = landmark_positions(V0, F, face_idx, bary)
    mask_bool = {}
    for k in masks.files:
        mb = np.zeros(len(V0), bool)
        mb[masks[k]] = True
        mask_bool[k] = mb
    on_mask = lambda k: mask_bool[k][F[face_idx]].any(1)
    e4c_bad = [i for i in LEFT_EYE_RING if not on_mask("left_eye_region")[i]]
    e4c_bad += [i for i in RIGHT_EYE_RING if not on_mask("right_eye_region")[i]]
    for i in e4c_bad:
        reasons[i].append("E4c eye ring off the eye region")
    e4d_pairs_bad = [(u, l) for u, l in zip(INNER_LIP_UPPER, INNER_LIP_LOWER) if Pt[u, 1] < Pt[l, 1]]
    for u, l in e4d_pairs_bad:
        reasons[u].append(f"E4d upper inner lip below {l}")
        reasons[l].append(f"E4d lower inner lip above {u}")
    e4d_off = [i for i in INNER_LIP_UPPER + INNER_LIP_LOWER if not on_mask("lips")[i]]
    for i in e4d_off:
        reasons[i].append("E4d inner lip off the lips")
    valid = np.array([not reasons[i] for i in range(468)])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, face_idx=face_idx, bary=bary.astype(np.float64), valid=valid, residual_mm=residual_mm)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["landmark", "face_idx", "b0", "b1", "b2", "valid", "residual_mm"])
        for i in range(468):
            w.writerow([i, int(face_idx[i]), repr(bary[i, 0]), repr(bary[i, 1]), repr(bary[i, 2]),
                        int(valid[i]), f"{residual_mm[i]:.4f}"])
    print(f"wrote {out.name} and {out.with_suffix('.csv').name}: {valid.sum()} valid of 468")

    run_checks(C, Fc, V0, F, masks, mask_bool, eyeball, face_idx, bary, valid, residual_mm, reasons,
               e4c_bad, e4d_pairs_bad, e4d_off)


def run_checks(C, Fc, V0, F, masks, mask_bool, eyeball, face_idx, bary, valid, residual_mm, reasons,
               e4c_bad, e4d_pairs_bad, e4d_off):
    verdict = lambda ok: "PASS" if ok else "FAIL"

    # E1: format.
    rows = bary.sum(1)
    e1 = ((face_idx >= 0) & (face_idx < len(F))).all() and (bary >= 0).all() \
        and np.abs(rows - 1).max() <= 1e-12 and not eyeball[F[face_idx]].any()
    print(f"E1 format: {verdict(e1)} (face_idx in [{face_idx.min()}, {face_idx.max()}], min bary {bary.min():.3g}, "
          f"max |row sum - 1| {np.abs(rows - 1).max():.2e}, eyeball triangles {int(eyeball[F[face_idx]].any(1).sum())})")

    # E2: residuals over valid landmarks.
    rv = residual_mm[valid]
    med, p95 = np.median(rv), np.percentile(rv, 95)
    print(f"E2 residual (valid, n={valid.sum()}): {verdict(med <= 1.5 and p95 <= 4.0)} median {med:.3f} mm, "
          f"95th pct {p95:.3f} mm, max {rv.max():.3f} mm")
    print(f"   invalid: {int((~valid).sum())}")
    for i in np.flatnonzero(~valid):
        print(f"     {i}: {'; '.join(reasons[i])}")
    order = np.argsort(-residual_mm)[:10]
    print("   10 largest residuals (all landmarks): " +
          ", ".join(f"{i} {residual_mm[i]:.2f}{'' if valid[i] else ' (invalid)'}" for i in order))

    # E3: symmetry on the undeformed template.
    mirror = C * np.array([-1.0, 1.0, 1.0])
    dm, partner = cKDTree(C).query(mirror)
    on_mid = partner == np.arange(len(C))
    print(f"E3 canonical mirror pairs: {int((dm <= 0.0005).sum())} of 468 vertices have a mirror partner within "
          f"0.5 mm (largest {dm.max() * 1000:.3f} mm); {int(on_mid.sum())} on the midline; "
          f"partner relation symmetric for {int((partner[partner] == np.arange(len(C))).sum())}")
    Pt = landmark_positions(V0, F, face_idx, bary)
    x_mid = Pt[on_mid & valid, 0].mean()
    pairs = [(i, int(partner[i])) for i in range(len(C)) if i < partner[i]]
    dist = np.array([np.linalg.norm(Pt[i] * [-1, 1, 1] + [2 * x_mid, 0, 0] - Pt[j]) for i, j in pairs]) * 1000
    both = np.array([valid[i] and valid[j] for i, j in pairs])
    dv = dist[both]
    print(f"E3 template symmetry (midline x = mean x of the valid midline landmarks = {x_mid * 1000:.3f} mm): "
          f"{verdict(np.median(dv) <= 1.5 and dv.max() <= 5.0)} valid pairs n={both.sum()}: median "
          f"{np.median(dv):.3f} mm, max {dv.max():.3f} mm; all pairs n={len(pairs)}: median {np.median(dist):.3f} mm, "
          f"max {dist.max():.3f} mm")
    worst = np.argsort(-dist)[:5]
    print("   5 worst pairs: " + ", ".join(
        f"{pairs[k][0]}/{pairs[k][1]} {dist[k]:.2f} mm{'' if both[k] else ' (invalid)'}" for k in worst))

    # E4: semantics on the undeformed template.
    nose, lips = masks["nose"], masks["lips"]
    tip = nose[np.argmax(V0[nose, 2])]
    d_a = np.linalg.norm(Pt[NOSE_TIP] - V0[tip]) * 1000
    lip_r, lip_l = lips[np.argmin(V0[lips, 0])], lips[np.argmax(V0[lips, 0])]
    d_r = np.linalg.norm(Pt[MOUTH_RIGHT] - V0[lip_r]) * 1000
    d_l = np.linalg.norm(Pt[MOUTH_LEFT] - V0[lip_l]) * 1000
    print(f"E4a nose tip: {verdict(d_a <= 5)} landmark 1 is {d_a:.2f} mm from the most forward nose vertex")
    print(f"E4b mouth corners: {verdict(d_r <= 6 and d_l <= 6)} 61 {d_r:.2f} mm, 291 {d_l:.2f} mm from the lips "
          f"vertices with the smallest and largest x")
    print(f"E4c eye rings: {verdict(not e4c_bad)} {32 - len(e4c_bad)} of 32 on a triangle with an eye-region vertex"
          f"{'; off: ' + str(e4c_bad) if e4c_bad else ''}")
    gaps = [(u, l, (Pt[u, 1] - Pt[l, 1]) * 1000) for u, l in zip(INNER_LIP_UPPER, INNER_LIP_LOWER)]
    print(f"E4d inner lips: pairs with the upper point below the lower: {len(e4d_pairs_bad)} of 9; off the lips: "
          f"{len(e4d_off)} of 18{' ' + str(e4d_off) if e4d_off else ''} (such landmarks are marked invalid above)")
    print("   upper - lower y (mm): " + ", ".join(f"{u}/{l} {g:+.2f}" for u, l, g in gaps))
    for k in ("left_ear", "right_ear", "neck", "scalp", "boundary"):
        ids = np.flatnonzero(mask_bool[k][F[face_idx]].any(1))
        print(f"E4e {k}: {len(ids)} landmarks {ids.tolist() if len(ids) else ''}")


if __name__ == "__main__":
    main()
