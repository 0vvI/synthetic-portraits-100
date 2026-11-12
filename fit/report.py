"""Print the tables of fit/README.md as Markdown, from fit/results/summary.json and solver_agreement.json.

    python fit/report.py [--results fit/results]

Every number in the write-up is pasted from this output, so the text cannot drift from the files.
"""
import argparse
import json
from pathlib import Path

import problem

WHAT = {"M0": "mean face, camera pose only", "M1": "+ shape", "M2": "+ expression",
        "M3": "+ one focal length for all portraits", "M4": "+ landmark offsets learned from other portraits",
        "M0+offsets": "control: mean face with the M4 offsets"}
REGION = {"face_oval": "face outline", "lips": "lips", "nose": "nose", "eyes": "eyes", "brows": "brows"}


def spread(q):
    return f"{q['median']:.2f} ({q['q1']:.2f} to {q['q3']:.2f})"


def setting_text(s):
    parts = [f"{s['focal_mm']:g} mm"]
    if s.get("n_shape"):
        parts.append(f"{s['n_shape']} shape, prior {s['p_shape']:g}")
    if s.get("n_expr"):
        parts.append(f"{s['n_expr']} expression, prior {s['p_expr']:g}")
    return "; ".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(problem.RESULTS))
    results = Path(ap.parse_args().results)
    d = json.loads((results / "summary.json").read_text())
    lm = d["landmarks"]
    print(f"{d['n_portraits']} portraits, {lm['valid']} valid landmarks: {lm['fit']} fit, {lm['val']} validation, "
          f"{lm['test']} test. Inter-ocular distance {spread(d['interocular_px'])} px. Solver: {d['solver']}"
          f"{' (quick run)' if d.get('quick') else ''}; {d['n_solves']} solves in {d['seconds']} s.\n")

    print("| Stage | What it adds | Chosen setting | Error on test landmarks, median (quartiles) | "
          "Largest shape coefficient, median | Portraits with a coefficient beyond 3 |")
    print("|---|---|---|---|---|---|")
    for stage, s in d["stages"].items():
        print(f"| {stage} | {WHAT[stage]} | {setting_text(s['setting'])} | {spread(s['test'])} | "
              f"{s['largest_shape_coef']['median']:.2f} | {round(100 * s['share_portraits_beyond_3'])} % |")
    print()
    for stage, s in d["stages"].items():
        print(f"{stage}: settings tried {s['settings']['tried']}, eligible {s['settings']['eligible']}; "
              f"validation {spread(s['val'])}; fit {spread(s['fit'])}; not converged {s['not_converged']}")
    print("\nPortraits better on test landmarks:", d["portraits_better_on_test"], "\n")

    stages = list(d["stages"])
    print("| Region left out | Landmarks | " + " | ".join(stages) + " |")
    print("|---|---|" + "---|" * len(stages))
    for region, r in d["region_test"].items():
        print(f"| {REGION[region]} | {r['n_landmarks']} | " + " | ".join(f"{r[s]['median']:.2f}" for s in stages) + " |")
    print()

    print("| Focal length (mm) | Validation error, median (quartiles) | Portraits where the chosen one is better |")
    print("|---|---|---|")
    for fm, v in d["focal_sweep_mm"].items():
        print(f"| {fm} | {spread(v['val'])} | {v['portraits_where_chosen_is_better']} |")
    print()

    o = d["landmark_offsets"]
    print(f"Offsets: median {o['size_mm']['median']:.2f} mm, 95th percentile {o['size_mm']['pct95']:.2f} mm, largest "
          f"{o['size_mm']['max']:.2f} mm; eye landmarks median {o['size_mm']['eyes_median']:.2f} mm. Two halves: "
          f"correlation {o['two_halves']['correlation']:.3f}, mean difference {o['two_halves']['mean_difference_mm']:.3f} "
          f"mm against a mean size of {o['two_halves']['mean_size_mm']:.3f} mm.")
    ff = d["final_fit"]
    print(f"Final fit (all landmarks, {setting_text(ff['setting'])}): error {spread(ff['error_all_landmarks'])}; "
          f"largest shape coefficient {spread(ff['largest_shape_coef'])}, max {ff['largest_shape_coef_max']:.2f}; "
          f"largest expression coefficient max {ff['largest_expr_coef_max']:.2f}; portraits beyond 3: "
          f"{round(100 * ff['share_portraits_beyond_3'])} %; camera distance {spread(ff['distance_m'])} m.\n")

    builds = {"C++ core, Eigen only": results / "solver_agreement.json",
              "C++ core, Eigen over a BLAS library (`-DFITCORE_BLAS=ON`)": results / "solver_agreement_blas.json"}
    rows = []
    for name, path in builds.items():
        if not path.exists():
            continue
        g = json.loads(path.read_text())
        ld = g["largest_difference"]
        print(f"{name} against numpy on {g['problems']} real problems ({g['landmarks_used']} landmarks, "
              f"{g['coefficients']} coefficients): largest difference in theta {ld['theta']:.1e}, in R {ld['R']:.1e}, "
              f"in t {ld['t_metres']:.1e} m, in cost {ld['cost_relative']:.1e} (relative); same status on "
              f"{g['same_status']}, same number of steps on {g['same_iteration_count']}; median steps "
              f"{g['iterations_median']:g}; not converged: numpy {g['not_converged']['numpy']}, "
              f"C++ {g['not_converged']['cpp']}.")
        ms, loop, batch = g["ms_per_solve_median"], g["loop_ms"], g["batch_100_ms"]
        if not rows:
            rows.append(f"| numpy reference ({g['machine']['numpy_blas']} BLAS) | {ms['numpy']:.1f} ms | "
                        f"{loop['numpy'] / 1000:.2f} s | |")
        rows.append(f"| {name} | {ms['cpp']:.1f} ms | {loop['cpp'] / 1000:.2f} s | "
                    f"{batch['all_threads'] / 1000:.2f} s |")
    if rows:
        print("\n| Solver | One fit, median | 100 fits, one after another | 100 fits, one batch on all cores |")
        print("|---|---|---|---|")
        print("\n".join(rows))


if __name__ == "__main__":
    main()
