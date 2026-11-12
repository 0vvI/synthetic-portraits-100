"""Fit FLAME to the landmarks of the 100 portraits, stage by stage, and evaluate each stage (SPEC.md 1 and 8).

    python fit/run_fit.py [--solver auto|cpp|ref] [--quick] [--out DIR]

Stages: M0 mean face and camera pose; M1 adds shape; M2 adds expression; M3 picks one focal length for all
portraits; M4 adds per-landmark offsets that correct where the detector's landmarks sit on the head model.

Two rules keep the numbers honest.
- Every setting (how many coefficients, how strong the prior, which focal length) is chosen on the validation
  landmarks and reported on the test landmarks, which a portrait's own fit never sees. A setting can only be
  chosen if it keeps the faces plausible: at most 5 % of the portraits may get a coefficient beyond 3 standard
  deviations. Without that rule the validation landmarks, which sit between fitted ones, reward extreme
  faces.
- The M4 offsets are learned from other portraits. The portraits are split into two halves; each half is
  fitted with offsets learned from the other half only. Learning uses all landmarks of those other portraits,
  so the M4 test error is held out across portraits, not across landmark ids.

The full run tries about 700 settings and is meant for the C++ solver (fit/fitcore). --quick tries about 60
and suits the numpy solver; send its output somewhere else with --out so it does not replace the full results.

Writes, under fit/results/:
  selection.csv          every setting tried, with its fit and validation error
  focal_sweep.csv        per portrait: validation error at each focal length of the M3 ladder
  stages.csv             per portrait and stage: fit, validation and test error
  region_test.csv        per portrait, stage and face region: error on a region the fit did not see
  landmark_offsets.npz   the M4 offsets: learned from each half, and from all portraits (used by the pictures)
  fits_final.npz         pose and coefficients per portrait, fitted on all valid landmarks with the M4 setting
  summary.json           the chosen settings and the numbers quoted in the README
Errors are in percent of the portrait's inter-ocular distance.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import problem

FOCAL_DEFAULT_MM = 85.0                                   # M0 to M2: a portrait lens
FOCAL_LADDER_MM = (24.0, 35.0, 50.0, 70.0, 85.0, 105.0, 135.0, 200.0, 400.0)
N_SHAPE_LADDER = (50, 100, 200)
N_EXPR_LADDER = (10, 20, 50)
# Prior weights, in percent of inter-ocular distance per unit coefficient. With the offsets the landmark
# residuals are about three times smaller, so M4 searches a lower range.
PRIOR_LADDER = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PRIOR_LADDER_M4 = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
QUICK = dict(n_shape=(50, 100), n_expr=(20, 50), prior=(1.0, 2.0, 4.0), prior_m4=(0.5, 1.0, 2.0))
N_SHAPE_MAX, N_EXPR_MAX = 200, 50
COEF_LIMIT = 3.0                                          # standard deviations
MAX_SHARE_BEYOND_LIMIT = 0.05                             # of the portraits, for a setting to be eligible
HALVES_SEED = 20261112
OFFSET_ROUNDS = 3
M = problem.N_LANDMARKS


class Fitter:
    """Holds the data once and fits all portraits for one setting at a time."""

    def __init__(self, solver_name):
        self.ids, self.xy, (self.W, self.H) = problem.load_landmarks()
        self.regions = problem.load_regions()
        self.valid = problem.load_valid()
        self.iod = problem.interocular(self.xy, self.regions)
        self.P0, self.B_all = problem.landmark_model(N_SHAPE_MAX, N_EXPR_MAX)
        self.cx, self.cy = self.W / 2.0, self.H / 2.0
        self.solver = problem.Solver(solver_name)
        self.N = len(self.ids)
        self.n_solves = 0

    def basis(self, n_shape, n_expr):
        cols = np.r_[0:n_shape, N_SHAPE_MAX:N_SHAPE_MAX + n_expr]
        return np.ascontiguousarray(self.B_all[:, :, cols])

    def groups(self, offsets):
        """[(portrait indices, P0)]: one landmark model for everyone, or one per group of portraits.

        offsets is None or a list of (portrait indices, delta), delta (468, 2) being offsets in metres along
        the model's x and y that are added to the landmark positions.
        """
        if offsets is None:
            return [(np.arange(self.N), self.P0)]
        return [(idx, self.P0 + np.column_stack([delta, np.zeros(M)])) for idx, delta in offsets]

    def fit(self, use, focal_mm, n_shape=0, n_expr=0, p_shape=0.0, p_expr=0.0, offsets=None):
        """Fit the portraits on the landmarks in the mask `use`. Returns (f, B, results).

        Landmark weights are 100 / inter-ocular distance, so residuals are in percent of that distance and
        the prior weights mean the same for every portrait. The pose is fitted first with the mean face; the
        coefficients then start at zero from that pose. results[i] is None for a portrait in no group.
        """
        f = problem.focal_px(focal_mm, self.W, self.H)
        use = use & self.valid
        B0, B = self.basis(0, 0), self.basis(n_shape, n_expr)
        p = np.r_[np.full(n_shape, p_shape), np.full(n_expr, p_expr)]
        results = [None] * self.N
        for idx, P0 in self.groups(offsets):
            xs = self.xy[idx]
            ws = np.where(use[None, :], 100.0 / self.iod[idx, None], 0.0)
            poses = [problem.initial_pose(P0, x, use, f, self.cx, self.cy) for x in xs]
            Rs, ts = np.stack([q[0] for q in poses]), np.stack([q[1] for q in poses])
            res = self.solver.solve_many(P0, B0, xs, ws, np.zeros(0), f, self.cx, self.cy, Rs, ts,
                                         np.zeros((len(idx), 0)))
            self.n_solves += len(idx)
            if n_shape + n_expr:
                Rs, ts = np.stack([r["R"] for r in res]), np.stack([r["t"] for r in res])
                res = self.solver.solve_many(P0, B, xs, ws, p, f, self.cx, self.cy, Rs, ts,
                                             np.zeros((len(idx), n_shape + n_expr)))
                self.n_solves += len(idx)
            for i, r in zip(idx, res):
                results[i] = r
        return f, (B if n_shape + n_expr else B0), results

    def residuals(self, f, B, results, offsets=None):
        """Predicted minus observed landmark pixels, (N, 468, 2); rows of portraits without a result are nan."""
        out = np.full((self.N, M, 2), np.nan)
        for idx, P0 in self.groups(offsets):
            for i in idx:
                r = results[i]
                out[i] = problem.project(P0, B, f, self.cx, self.cy, r["R"], r["t"], r["theta"]) - self.xy[i]
        return out

    def errors(self, residuals, mask):
        """Per-portrait mean landmark distance over the valid landmarks of `mask`, in percent of the
        inter-ocular distance."""
        mask = mask & self.valid
        return np.linalg.norm(residuals[:, mask], axis=2).mean(1) / self.iod * 100.0

    def learn_offsets(self, idx, setting):
        """Per-landmark offsets (468, 2), metres along the model's x and y, learned from the portraits idx.

        Each round fits those portraits on all their valid landmarks with the current offsets, then moves
        every landmark by the in-plane step that best removes its remaining residual over those portraits
        (a two-parameter least-squares problem per landmark). A frontal still says nothing about a landmark's
        depth, so the offsets have no z component.
        """
        delta = np.zeros((M, 2))
        everything = np.ones(M, bool)
        for _ in range(OFFSET_ROUNDS):
            f, B, results = self.fit(everything, offsets=[(idx, delta)], **setting)
            P0 = self.P0 + np.column_stack([delta, np.zeros(M)])
            A, b = np.zeros((M, 2, 2)), np.zeros((M, 2))
            for i in idx:
                r = results[i]
                Q = (P0 + B @ r["theta"]) @ r["R"].T + r["t"]
                seen = np.stack([f * Q[:, 0] / Q[:, 2] + self.cx, f * Q[:, 1] / Q[:, 2] + self.cy], axis=1)
                e = (self.xy[i] - seen) * (100.0 / self.iod[i])
                s = f / Q[:, 2] * (100.0 / self.iod[i])
                D = np.zeros((M, 2, 3))                      # d(pixel) / d(camera point), as in SPEC.md 3
                D[:, 0, 0] = D[:, 1, 1] = s
                D[:, 0, 2], D[:, 1, 2] = -s * Q[:, 0] / Q[:, 2], -s * Q[:, 1] / Q[:, 2]
                J = D @ r["R"][:, :2]                        # effect of a step along the model's x and y
                A += np.einsum("mij,mik->mjk", J, J)
                b += np.einsum("mij,mi->mj", J, e)
            step = np.linalg.solve(A, b[..., None])[..., 0]
            step[~self.valid] = 0.0
            delta = delta + step
        return delta


def largest_coefficients(results, n_shape):
    """Per portrait: the largest absolute shape coefficient and the largest absolute expression coefficient."""
    shape = np.array([np.abs(r["theta"][:n_shape]).max(initial=0.0) for r in results])
    expr = np.array([np.abs(r["theta"][n_shape:]).max(initial=0.0) for r in results])
    return shape, expr


def quartiles(a):
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"median": round(float(med), 4), "q1": round(float(q1), 4), "q3": round(float(q3), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solver", default="auto", choices=("auto", "cpp", "ref"))
    ap.add_argument("--quick", action="store_true", help="coarse ladders (about 60 settings)")
    ap.add_argument("--out", default=str(problem.RESULTS), help="output directory")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_shapes, n_exprs, priors, priors_m4 = N_SHAPE_LADDER, N_EXPR_LADDER, PRIOR_LADDER, PRIOR_LADDER_M4
    if a.quick:
        n_shapes, n_exprs, priors, priors_m4 = (QUICK[k] for k in ("n_shape", "n_expr", "prior", "prior_m4"))

    t_start = time.perf_counter()
    F = Fitter(a.solver)
    masks = problem.split_masks()
    n_used = {k: int((m & F.valid).sum()) for k, m in masks.items()}
    print(f"solver: {F.solver.name}; landmarks fit/val/test: {n_used} (of {int(F.valid.sum())} valid)")

    selection = []           # rows of selection.csv
    counts = {}              # stage -> how many settings were tried and how many were eligible
    focal_val = {}           # focal length -> per-portrait validation error, for the M3 ladder
    stage_fits = {}          # stage -> (setting, offsets, f, B, results)

    def run(stage, offsets, setting):
        f, B, res = F.fit(masks["fit"], offsets=offsets, **setting)
        resid = F.residuals(f, B, res, offsets)
        e_fit, e_val = F.errors(resid, masks["fit"]), F.errors(resid, masks["val"])
        top_shape, top_expr = largest_coefficients(res, setting.get("n_shape", 0))
        share = float(np.mean(np.maximum(top_shape, top_expr) > COEF_LIMIT))
        row = dict(stage=stage, focal_mm=setting["focal_mm"], n_shape=setting.get("n_shape", 0),
                   p_shape=setting.get("p_shape", 0.0), n_expr=setting.get("n_expr", 0),
                   p_expr=setting.get("p_expr", 0.0), fit_median=float(np.median(e_fit)),
                   val_median=float(np.median(e_val)), val_q1=float(np.percentile(e_val, 25)),
                   val_q3=float(np.percentile(e_val, 75)),
                   largest_shape_coef_median=float(np.median(top_shape)),
                   largest_expr_coef_median=float(np.median(top_expr)),
                   share_beyond_3=share, eligible=int(share <= MAX_SHARE_BEYOND_LIMIT),
                   not_converged=sum(r["status"] != "converged" for r in res), chosen=0)
        selection.append(row)
        if stage == "M3":
            focal_val[setting["focal_mm"]] = e_val
        return row, (setting, offsets, f, B, res)

    def choose(stage, settings, offsets=None):
        """Run all settings; among the eligible ones keep the lowest median validation error."""
        best, n_eligible = None, 0
        for setting in settings:
            row, fit = run(stage, offsets, setting)
            n_eligible += row["eligible"]
            if row["eligible"] and (best is None or row["val_median"] < best[0]["val_median"]):
                best = (row, fit)
        if best is None:
            raise SystemExit(f"{stage}: no setting keeps the coefficients within the limit")
        row, fit = best
        row["chosen"] = 1
        stage_fits[stage] = fit
        counts[stage] = dict(tried=len(settings), eligible=n_eligible)
        print(f"{stage}: {n_eligible} of {len(settings)} settings eligible; chose {fit[0]} -> fit "
              f"{row['fit_median']:.3f}, validation {row['val_median']:.3f}, share beyond 3: "
              f"{row['share_beyond_3']:.2f}", flush=True)
        return fit[0]

    def grid(focal_mm, prior_ladder):
        return [dict(focal_mm=focal_mm, n_shape=ns, p_shape=ps, n_expr=ne, p_expr=pe)
                for ns in n_shapes for ps in prior_ladder for ne in n_exprs for pe in prior_ladder]

    choose("M0", [dict(focal_mm=FOCAL_DEFAULT_MM)])
    choose("M1", [dict(focal_mm=FOCAL_DEFAULT_MM, n_shape=n, p_shape=p) for n in n_shapes for p in priors])
    m2 = choose("M2", grid(FOCAL_DEFAULT_MM, priors))
    m3 = choose("M3", [dict(m2, focal_mm=fm) for fm in FOCAL_LADDER_MM])

    # M4: offsets learned from the other half of the portraits, at the M3 setting.
    order = np.random.default_rng(HALVES_SEED).permutation(F.N)
    halves = [np.sort(order[:F.N // 2]), np.sort(order[F.N // 2:])]
    learned = [F.learn_offsets(h, m3) for h in halves]
    crossed = [(halves[0], learned[1]), (halves[1], learned[0])]
    offsets_all = F.learn_offsets(np.arange(F.N), m3)
    size_mm = np.linalg.norm(offsets_all, axis=1)[F.valid] * 1000.0
    in_eyes = np.zeros(M, bool)
    in_eyes[F.regions["eyes"]] = True
    offsets_summary = dict(
        rounds=OFFSET_ROUNDS, halves_seed=HALVES_SEED, learned_at=m3,
        size_mm=dict(median=round(float(np.median(size_mm)), 3), pct95=round(float(np.percentile(size_mm, 95)), 3),
                     max=round(float(size_mm.max()), 3),
                     eyes_median=round(float(np.median(np.linalg.norm(offsets_all, axis=1)[in_eyes & F.valid]))
                                       * 1000.0, 3)),
        two_halves=dict(
            correlation=round(float(np.corrcoef(learned[0][F.valid].ravel(), learned[1][F.valid].ravel())[0, 1]), 4),
            mean_difference_mm=round(float(np.linalg.norm(learned[0] - learned[1], axis=1)[F.valid].mean())
                                     * 1000.0, 3),
            mean_size_mm=round(float(np.linalg.norm(learned[0], axis=1)[F.valid].mean()) * 1000.0, 3)))
    print(f"offsets: median {offsets_summary['size_mm']['median']} mm, max {offsets_summary['size_mm']['max']} mm; "
          f"the two halves agree with correlation {offsets_summary['two_halves']['correlation']}", flush=True)
    choose("M0+offsets", [dict(focal_mm=m3["focal_mm"])], crossed)      # a control, not a stage
    m4 = choose("M4", grid(m3["focal_mm"], priors_m4), crossed)

    # Test error per stage: computed once, after every choice is made.
    stage_rows, summary_stages, test_error = [], {}, {}
    for stage, (setting, offsets, f, B, res) in stage_fits.items():
        resid = F.residuals(f, B, res, offsets)
        e = {k: F.errors(resid, masks[k]) for k in ("fit", "val", "test")}
        test_error[stage] = e["test"]
        top, top_expr = largest_coefficients(res, setting.get("n_shape", 0))
        for i, pid in enumerate(F.ids):
            stage_rows.append([pid, stage, f"{e['fit'][i]:.4f}", f"{e['val'][i]:.4f}", f"{e['test'][i]:.4f}",
                               f"{top[i]:.4f}", f"{top_expr[i]:.4f}", res[i]["iterations"], res[i]["status"]])
        summary_stages[stage] = dict(setting=setting, focal_px=round(f, 2), settings=counts[stage],
                                     fit=quartiles(e["fit"]), val=quartiles(e["val"]), test=quartiles(e["test"]),
                                     largest_shape_coef=quartiles(top), largest_expr_coef=quartiles(top_expr),
                                     share_portraits_beyond_3=float(np.mean(np.maximum(top, top_expr) > COEF_LIMIT)),
                                     not_converged=sum(r["status"] != "converged" for r in res))
        print(f"{stage}: test median {np.median(e['test']):.3f} (quartiles {np.percentile(e['test'], 25):.3f} to "
              f"{np.percentile(e['test'], 75):.3f}); largest |shape coefficient| median {np.median(top):.2f}")
    better = {f"{s}_than_M0": int(np.sum(test_error[s] < test_error["M0"])) for s in ("M1", "M2", "M3", "M4")}
    better["M4_than_M3"] = int(np.sum(test_error["M4"] < test_error["M3"]))
    better["M4_than_M0+offsets"] = int(np.sum(test_error["M4"] < test_error["M0+offsets"]))

    # Region test: leave one face region out of the fit and measure the error on it.
    everything = np.ones(M, bool)
    region_rows, summary_regions = [], {}
    for region in problem.REGIONS:
        inside = np.zeros(M, bool)
        inside[F.regions[region]] = True
        summary_regions[region] = {"n_landmarks": int((inside & F.valid).sum())}
        for stage, (setting, offsets, _, _, _) in stage_fits.items():
            f, B, res = F.fit(everything & ~inside, offsets=offsets, **setting)
            e = F.errors(F.residuals(f, B, res, offsets), inside)
            summary_regions[region][stage] = quartiles(e)
            region_rows += [[pid, region, stage, f"{e[i]:.4f}"] for i, pid in enumerate(F.ids)]
        print(f"region {region}: " + ", ".join(f"{s} {summary_regions[region][s]['median']:.2f}"
                                               for s in stage_fits), flush=True)

    # The fit that the pictures use: all valid landmarks, the M4 setting, offsets learned from all portraits.
    final_offsets = [(np.arange(F.N), offsets_all)]
    f, B, res = F.fit(everything, offsets=final_offsets, **m4)
    n_shape = m4["n_shape"]
    theta = np.stack([r["theta"] for r in res])
    e_all = F.errors(F.residuals(f, B, res, final_offsets), everything)
    top, top_expr = largest_coefficients(res, n_shape)
    np.savez(out / "fits_final.npz", ids=F.ids, R=np.stack([r["R"] for r in res]),
             t=np.stack([r["t"] for r in res]), shape=theta[:, :n_shape], expr=theta[:, n_shape:],
             focal_px=f, cx=F.cx, cy=F.cy, error_pct_iod=e_all)
    np.savez(out / "landmark_offsets.npz", half_a=halves[0], half_b=halves[1], learned_from_a=learned[0],
             learned_from_b=learned[1], learned_from_all=offsets_all, valid=F.valid)

    with open(out / "selection.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        keys = list(selection[0])
        w.writerow(keys)
        for row in selection:
            w.writerow([f"{row[k]:.4f}" if k.endswith(("median", "q1", "q3", "beyond_3")) else row[k] for k in keys])
    with open(out / "focal_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id"] + [f"val_error_{fm:g}mm" for fm in focal_val])
        for i, pid in enumerate(F.ids):
            w.writerow([pid] + [f"{e[i]:.4f}" for e in focal_val.values()])
    chosen_val = focal_val[m3["focal_mm"]]
    focal_summary = {f"{fm:g}": dict(val=quartiles(e), portraits_where_chosen_is_better=int(np.sum(chosen_val < e)))
                     for fm, e in focal_val.items()}
    with open(out / "stages.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "stage", "fit_error", "val_error", "test_error", "largest_shape_coef",
                    "largest_expr_coef", "iterations", "status"])
        w.writerows(stage_rows)
    with open(out / "region_test.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["id", "region", "stage", "error"])
        w.writerows(region_rows)
    summary = dict(
        unit="percent of inter-ocular distance",
        solver=F.solver.name, quick=bool(a.quick), n_portraits=F.N, image_size=[F.W, F.H],
        landmarks=dict(valid=int(F.valid.sum()), **n_used), split_seed=problem.SPLIT_SEED,
        interocular_px=quartiles(F.iod), stages=summary_stages,
        portraits_better_on_test=better, focal_sweep_mm=focal_summary, landmark_offsets=offsets_summary,
        region_test=summary_regions,
        selection_rule=dict(coefficient_limit=COEF_LIMIT, max_share_of_portraits_beyond_it=MAX_SHARE_BEYOND_LIMIT),
        final_fit=dict(setting=m4, offsets="learned from all portraits",
                       error_all_landmarks=quartiles(e_all), largest_shape_coef=quartiles(top),
                       largest_shape_coef_max=round(float(top.max()), 4),
                       largest_expr_coef=quartiles(top_expr),
                       largest_expr_coef_max=round(float(top_expr.max()), 4),
                       share_portraits_beyond_3=float(np.mean(np.maximum(top, top_expr) > COEF_LIMIT)),
                       distance_m=quartiles(np.array([r["t"][2] for r in res]))),
        n_solves=F.n_solves, seconds=round(time.perf_counter() - t_start, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"final fit on all landmarks: error {np.median(e_all):.3f}; {F.n_solves} solves in "
          f"{summary['seconds']} s with the {F.solver.name} solver")


if __name__ == "__main__":
    main()
