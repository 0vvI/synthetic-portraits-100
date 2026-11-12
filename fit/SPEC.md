# Head-shape fit: the contract

This file fixes the maths, the file layout and the tests for the head-shape fit, so that the numpy reference
solver and the C++ solver can be written separately and must agree. If code and this file disagree, the code
is wrong or this file gets a dated correction; they are never left different.

## 1. What is fitted

For each portrait: FLAME 2023 Open shape and expression coefficients and one camera pose, fitted to the 468
face-mesh landmarks that MediaPipe Face Landmarker finds in the image. There is one still per identity, so
depth is weakly constrained; the evaluation says how weakly instead of hiding it.

Stages land one at a time. Each stage adds one row to the results table and must not break the earlier rows.

| Stage | What it adds |
|---|---|
| M0 | mean face (all coefficients zero), camera pose only: the baseline |
| M1 | shape coefficients with a Gaussian prior |
| M2 | expression coefficients |
| M3 | one focal length shared by all 100 portraits, chosen on validation landmarks |
| M4 | per-landmark offsets that correct where the detector's landmarks sit on the head model, learned from other portraits |
| later | jaw rotation, silhouette term, iris size as a depth cue, a test against known shapes |

## 2. Conventions

- Image pixels: x right, y down. Landmark pixel = MediaPipe's normalised coordinate times (W, H). Full-size
  portraits are 1792 x 2400.
- FLAME model space: metres, +x the subject's left, +y up, +z out of the face.
- Camera space: +x right, +y down, +z forward (away from the camera).
- `R_FLIP = diag(1, -1, -1)` turns model axes into camera axes for a head that faces the camera. The Python
  wrapper applies it once to the landmark model. **The solvers never see FLAME axes.**
- All arrays are float64 and C-contiguous. M = number of landmarks, K = number of coefficients.

## 3. The problem the solvers get

Inputs:

| Name | Shape | Meaning |
|---|---|---|
| `P0` | (M, 3) | landmark positions at theta = 0, camera axes, metres |
| `B` | (M, 3, K) | linear basis: `P_i(theta) = P0[i] + B[i] @ theta` |
| `x` | (M, 2) | observed landmark pixels |
| `w` | (M,) | landmark weights, >= 0; 0 leaves a landmark out (held-out points) |
| `p` | (K,) | prior weights; the prior residual is `p[k] * theta[k]` |
| `f, cx, cy` | scalars | focal length and principal point, pixels |

State: rotation `R` (3x3, in SO(3)), translation `t` (3,), coefficients `theta` (K,). K may be 0: then `B`
has shape (M, 3, 0), `p` and `theta` are empty, and the solve is a pose-only fit (stage M0).

```
Q_i = R @ P_i(theta) + t
u_i = f * Q_i.x / Q_i.z + cx
v_i = f * Q_i.y / Q_i.z + cy
```

Residual vector `r`, length 2M + K:

```
r[2i]     = w[i] * (u_i - x[i,0])
r[2i + 1] = w[i] * (v_i - x[i,1])
r[2M + k] = p[k] * theta[k]
cost      = 0.5 * r @ r
```

Jacobian `J`, shape (2M + K, 6 + K). Columns: `[d_omega (3), d_t (3), d_theta (K)]`. The rotation is perturbed
on the left: `R <- exp_so3(d_omega) @ R`, and the derivative is taken at `d_omega = 0`.

```
D_i = (f / Q_i.z) * [[1, 0, -Q_i.x / Q_i.z],
                     [0, 1, -Q_i.y / Q_i.z]]            # 2 x 3
rows of landmark i = w[i] * D_i @ [ -skew(Q_i - t) | I_3 | R @ B[i] ]
prior rows         = [ 0 | 0 | diag(p) ]
```

`skew(a)` is the matrix with `skew(a) @ b = cross(a, b)`.

`exp_so3(omega)`: with `a = |omega|` and `W = skew(omega)`; if `a < 1e-12` return `I + W`, else
`I + (sin a / a) W + ((1 - cos a) / a^2) W @ W`.

## 4. The solver (Levenberg-Marquardt), identical in both implementations

```
lam = lam0
r, J = residuals_and_jacobian(state);  cost = 0.5 * r @ r
status = "max_iter"
for it in 1 .. max_iter:
    A = J.T @ J;  g = J.T @ r
    accepted = False
    while True:
        Ad = A + lam * diag(max(diag(A), 1e-12))
        delta = solve(Ad, -g)                      # Cholesky / LDLT; Ad is symmetric positive definite
        cand  = (exp_so3(delta[0:3]) @ R,  t + delta[3:6],  theta + delta[6:])
        cost_c = 0.5 * |residuals(cand)|^2
        if isfinite(cost_c) and cost_c < cost:
            rel = (cost - cost_c) / max(cost, 1e-300)
            state, cost = cand, cost_c
            lam = max(lam / 10, 1e-15)
            accepted = True
            break
        if max(abs(delta)) < xtol: status = "converged"; break   # rejected, and too small to matter
        lam = lam * 10
        if lam > 1e12: break
    if status == "converged": break
    if not accepted: status = "stalled"; break
    r, J = residuals_and_jacobian(state)
    if rel < ftol or max(abs(delta)) < xtol: status = "converged"; break
```

A rejected step smaller than `xtol` ends the solve as converged: at the minimum the cost stops changing in
its last digits, so "no lower cost found" there is the normal ending, not a failure. `stalled` is left for
real trouble (a cost that is not finite, or no decrease at any damping with steps that are not small).

Defaults: `max_iter = 100`, `lam0 = 1e-3`, `ftol = 1e-14`, `xtol = 1e-12`. `iterations` counts accepted
steps. No randomness anywhere. The fits of the portraits pass `max_iter = 200` and `ftol = 1e-12`
(`SOLVE_OPTIONS` in `fit/problem.py`); the tests use the defaults.

## 5. Interfaces

### 5.1 `fit/solver_ref.py` (numpy only)

```python
exp_so3(omega) -> R                                                        # (3,) -> (3, 3)
residuals(P0, B, x, w, p, f, cx, cy, R, t, theta) -> r                     # (2M + K,)
residuals_and_jacobian(P0, B, x, w, p, f, cx, cy, R, t, theta) -> (r, J)
solve(P0, B, x, w, p, f, cx, cy, R, t, theta,
      max_iter=100, lam0=1e-3, ftol=1e-14, xtol=1e-12) -> dict
# dict keys: "R" (3,3), "t" (3,), "theta" (K,), "cost" float, "iterations" int, "status" str
```

### 5.2 `fit/fitcore/` (C++17, Eigen 3.4 or newer, no other runtime dependency)

The build flag `FITCORE_BLAS` (off by default) lets Eigen hand its large matrix products to a BLAS library.
It changes speed and the last digits of a result, nothing else; every test must pass with and without it.

```cpp
namespace fitcore {
using MatX3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;
using MatX2 = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;
using MatXX = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Problem {
  MatX3 P0;            // (M, 3)
  MatXX B;             // (3M, K): row 3*i + c holds B[i, c, :]
  MatX2 x;             // (M, 2)
  Eigen::VectorXd w;   // (M)
  Eigen::VectorXd p;   // (K)
  double f = 0, cx = 0, cy = 0;
};
struct State   { Eigen::Matrix3d R; Eigen::Vector3d t; Eigen::VectorXd theta; };
struct Options { int max_iter = 100; double lam0 = 1e-3, ftol = 1e-14, xtol = 1e-12; };
enum class Status { Converged = 0, MaxIter = 1, Stalled = 2 };
struct Result  { State state; double cost = 0; int iterations = 0; Status status = Status::MaxIter; };

Eigen::Matrix3d exp_so3(const Eigen::Vector3d& omega);
void   residuals(const Problem&, const State&, Eigen::VectorXd& r);
void   residuals_and_jacobian(const Problem&, const State&, Eigen::VectorXd& r, MatXX& J);
Result solve(const Problem&, const State& init, const Options& = {});
}
```

Python module `fitcore` (pybind11) exposes `exp_so3`, `residuals`, `residuals_and_jacobian`, `solve` with the
**same argument order, defaults and return values as 5.1** (status as the strings `"converged"`, `"max_iter"`,
`"stalled"`), so a script can swap one for the other. It also exposes

```python
solve_batch(P0, B, xs, ws, p, f, cx, cy, Rs, ts, thetas,
            max_iter=100, lam0=1e-3, ftol=1e-14, xtol=1e-12, n_threads=0) -> list[dict]
# xs (N, M, 2), ws (N, M), Rs (N, 3, 3), ts (N, 3), thetas (N, K); one model, N portraits.
# Releases the GIL; n_threads = 0 means hardware concurrency. Results in input order, bit-identical to N
# separate solve() calls.
```

Build:

```
cmake -S fit/fitcore -B build/fitcore -DCMAKE_BUILD_TYPE=Release        # library + C++ tests
cmake --build build/fitcore -j && ctest --test-dir build/fitcore --output-on-failure
cmake -S fit/fitcore -B build/fitcore -DFITCORE_PYTHON=ON ...           # also builds the Python module
```

The Python module is written to `build/fitcore/`; scripts and tests find it through the `FITCORE_BUILD_DIR`
environment variable, default `build/fitcore` under the repository root.

## 6. Tests (none needs FLAME, the portraits or the network)

Synthetic problem, used by every test: M = 120, K = 12, f = 1500, cx = 320, cy = 240; `P0` uniform in a
0.2 m cube centred on the origin; `B` standard normal times 0.01; true rotation 0.3 rad about a random axis;
true `t = (0.02, -0.01, 0.6)`; true theta standard normal; `x` = exact projection (no noise).

| Id | Test | Pass when |
|---|---|---|
| T1 | `exp_so3` | `R.T @ R = I` and `det R = 1` to 1e-14 for angles 0, 1e-13, 1e-6, 1, 3; the two branches agree to 1e-12 at angle 1e-12 |
| T2 | Jacobian against central differences (h = 1e-6) at a perturbed state, with non-trivial `w` and `p` | max abs difference <= 1e-6 x max abs J |
| T3 | recovery: start from identity rotation, `t = (0, 0, 0.5)`, theta = 0, with `p = 0` | status converged; max abs error in R, t, theta <= 1e-8 |
| T4 | prior: `p = 0.5` | status converged; cost never increases; at the result `max abs (J.T @ r) <= 1e-6 x norm(J) x norm(r)` (Frobenius norm of J, 2-norm of r) |
| T5 | zero weights: 30 landmarks get `w = 0` and their `x` replaced by garbage | same result as the problem with those 30 rows removed, to 1e-10 |
| T6 | agreement, C++ against numpy on the same inputs (Python test, skipped when the module is not built) | `r` and `J`: max abs difference <= 1e-9 x max abs value; solutions: R, t, theta within 1e-8, cost within 1e-10 relative plus 1e-12 absolute (the exact-data cases end at rounding noise) |
| T7 | `solve_batch` against a loop of `solve` | identical results |

T1 to T5 exist twice: in C++ (`fit/fitcore/tests`, run by `ctest`) and in Python for the reference
(`fit/tests/test_solver_ref.py`, run by `pytest`). The Python synthetic generator is `fit/tests/synth.py`,
seeded with `numpy.random.default_rng(seed)`; the C++ one uses `std::mt19937_64`. They need not produce the
same numbers: T6 passes numpy arrays to both.

## 7. Landmark model

- `fit/results/landmarks_mp478.npz`: `ids` (100,), `xy` (100, 478, 2) pixels at full size, `z` (100, 478)
  MediaPipe's relative depth times W (not used by the fit), `size` (100, 2) as (W, H). Points 468 to 477 are
  the irises; the fit uses 0 to 467.
- `fit/embedding/mediapipe468_on_flame2023.npz`: where each of the 468 landmarks sits on the FLAME 2023
  surface. `face_idx` (468,) int32 indexes FLAME's triangle array; `bary` (468, 3), non-negative, rows sum
  to 1; `valid` (468,) bool; `residual_mm` (468,) the distance left after the alignment that produced it.
  Landmark position = `sum_j bary[i, j] * V[F[face_idx[i], j]]`.
- With FLAME's template `V0` (5023, 3), shape basis `S` (5023, 3, 300) and expression basis `E`
  (5023, 3, 100), the wrapper builds `P0` and `B` by applying the barycentric weights and `R_FLIP`; theta is
  the first `n_shape` shape coefficients followed by the first `n_expr` expression coefficients.
- Only landmarks with `valid` set are used, in fitting and in every error.
- `fit/results/landmark_offsets.npz` (stage M4): `learned_from_a`, `learned_from_b`, `learned_from_all`, each
  (468, 2), offsets in metres along x and y of the camera-axes landmark model, added to `P0[:, :2]`;
  `half_a`, `half_b` the portrait indices of the two halves; `valid`. The solvers are unchanged: an offset
  only changes the `P0` they are given.

## 8. Evaluation protocol

- Landmarks are split once, by index, with seed 20261111: 60 % fit, 20 % validation, 20 % test. The same
  split is used for every portrait. Settings (number of coefficients, prior strength, focal length) are
  chosen on validation landmarks only and reported on test landmarks.
- Landmark weights are 100 / inter-ocular distance, so residuals and prior weights are in percent of that
  distance. The principal point is the image centre. Focal lengths are quoted as 35 mm equivalents, matched
  on the image diagonal; M0 to M2 use 85 mm.
- A setting is eligible only if at most 5 % of the portraits get a coefficient (shape or expression) beyond
  3 in absolute value. Among the eligible settings the one with the lowest median validation error is
  chosen. Reason: validation landmarks lie between fitted ones, so validation error alone keeps rewarding
  weaker priors and more extreme faces.
- Settings tried (`fit/run_fit.py`): shape coefficients 50, 100, 200; expression coefficients 10, 20, 50;
  prior weights 1, 1.5, 2, 3, 4, 6 for M1 to M3 and 0.25, 0.5, 0.75, 1, 1.5, 2 for M4; M2 and M4 search
  shape and expression settings jointly; M3 tries 24, 35, 50, 70, 85, 105, 135, 200 and 400 mm with the M2
  setting; M4 keeps the M3 focal length.
- M4 offsets: the portraits are split once into two halves (seed 20261112). The offsets for one half are
  learned from the other half only: those portraits are fitted on all their valid landmarks with the M3
  setting, each landmark is moved along the model's x and y by the least-squares step that best removes its
  residual over those portraits, and this is repeated three times. A frontal still does not constrain a
  landmark's depth, so the offsets have no z component. Every M4 number comes from portraits that had no
  part in the offsets they are fitted with. The row "M0+offsets" (mean face with the same offsets) is a
  control: it shows how much of M4 is the offsets alone.
- The pictures use one more fit that is not scored: all valid landmarks, the M4 setting, offsets learned from
  all 100 portraits (`fit/results/fits_final.npz`).
- Error = distance in pixels between the projected and the observed landmark, divided by that portrait's
  inter-ocular distance (the distance between the mean of the right-eye and the mean of the left-eye contour
  landmarks), in percent. Reported as the median over the 100 portraits, with the quartiles.
- Region test: fit on everything except one region (face oval, lips, nose, eyes, brows), report the error on
  that region. Random splits of dense landmarks are close to interpolation; this one is not.
- Saturation: the largest absolute shape and expression coefficient per portrait, and the share of
  portraits with any coefficient beyond 3.
- C++ against numpy on the 100 real problems: largest difference in theta and in cost.
- Timing: median time per solve, numpy and C++, same problems, same machine. The C++ solve runs on one
  thread; numpy runs as installed, so the BLAS library under its matrix product may use more. The batch call
  is timed on one thread and on all cores. Both C++ builds are timed (Eigen only, and `FITCORE_BLAS`).

## 9. What is not in the repository

The FLAME model files and the MediaPipe model are not redistributed here. `fit/flame_io.py` turns your own
`flame2023_Open.pkl` (CC BY 4.0, from the FLAME site) into `fit/assets/flame_min.npz`; `fit/assets/` is
ignored by git. The landmark file, the embedding and the fitted coefficients are in the repository, so the
tables can be checked without the full-size portraits.

## 10. Corrections

- 12 Nov 2026, section 6, T6: the cost tolerance gained an absolute part (1e-12). The exact-data test
  problems end at rounding noise, where a purely relative bound cannot be met.
- 12 Nov 2026, section 4: a rejected step smaller than `xtol` now ends the solve as `converged`. Before, a
  solve that had reached the minimum could end as `stalled` in one implementation and `converged` in the
  other, depending on the last bit of the cost (found by test T6 with `p = 0.5`; the two solutions agreed
  to 1e-11).
- 12 Nov 2026, section 6, T4: the gradient bound is now relative to the size of J and r, and the status is
  checked. The old bound, `1e-6 x max(1, cost)`, sat below what the cost comparison can resolve in double
  precision (about `1.5e-8 x norm(J) x norm(r)`), so it passed or failed by luck.
- 12 Nov 2026, sections 5.2 and 8: the first timing showed the C++ core slower than numpy on the real
  problems (about 2.4 times slower per fit; a measurement of that day, not kept as a file). Nearly all the time is the product J^T J, which numpy hands to a
  tuned BLAS library. The core now forms only the lower triangle of that product over the landmark rows,
  factorizes with a blocked Cholesky routine, and can be built over the same BLAS (`FITCORE_BLAS`). The
  timing rule was reworded to say what is compared with what.
- 12 Nov 2026, section 8: the eligibility rule, the weights, the list of settings tried and the M4 protocol
  were written down after the first runs on the portraits. The rule and the protocol were fixed by looking
  at validation landmarks only; test landmarks were read once, at the end.
- 12 Nov 2026, sections 1 and 7: stage M4 (landmark offsets) added. Before it, a fit that left the eye
  landmarks out predicted them worse than the mean face did; the residuals were nearly the same on every
  portrait, which pointed at the landmark mapping, not at the faces.
