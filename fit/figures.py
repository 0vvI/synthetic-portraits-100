"""Figures for the head-shape fit, from the files that run_fit.py writes.

    python fit/figures.py [--results fit/results] [--out docs/fit] [--portraits portraits/identities]

  stage_error.png    error on unseen landmarks, stage by stage (one dot per portrait)
  region_test.png    error on a face region that was left out of the fit
  focal_length.png   validation error against the shared focal length
  offsets.png        the per-landmark offsets of stage M4, drawn on the mean face
  overlay.jpg        the fitted head drawn over ten portraits, one per prompted tone

The first three need only the CSV files. The last two also need fit/assets/flame_min.npz, and the overlay
needs the portraits (full-size ones if present, otherwise the 600 x 800 thumbnails in images/thumbs).
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import flame_io
import problem

INK, INK_2, INK_3 = "#111114", "#4a4a52", "#6b6b75"
AXIS, GRID, SURFACE = "#9a9aa3", "#e4e4e8", "#ffffff"
STAGES = ("M0", "M1", "M2", "M3", "M4")
STAGE_LABEL = {"M0": "M0\nmean face,\npose only", "M1": "M1\n+ shape", "M2": "M2\n+ expression",
               "M3": "M3\n+ shared\nfocal length", "M4": "M4\n+ landmark\noffsets"}
REGION_STAGES = {"M0": "M0: mean face", "M3": "M3: shape, expression, focal length",
                 "M4": "M4: + landmark offsets"}
# Three roles, three colours: the mean face is the neutral reference, the fit is blue, and the fit with
# landmark offsets is orange. The stage itself is read from the axis, not from the colour.
GREY, BLUE, ORANGE = "#8a8a93", "#2a78d6", "#eb6834"
STAGE_COLOR = {"M0": GREY, "M1": BLUE, "M2": BLUE, "M3": BLUE, "M4": ORANGE}
REGION_LABEL = {"face_oval": "face outline", "lips": "lips", "nose": "nose", "eyes": "eyes", "brows": "brows"}
UNIT = "% of the distance between the eyes"


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)


def stage_error(results, out):
    rows = read_csv(results / "stages.csv")
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=160)
    for k, stage in enumerate(STAGES):
        e = np.array([float(r["test_error"]) for r in rows if r["stage"] == stage])
        q1, med, q3 = np.percentile(e, [25, 50, 75])
        ax.scatter(k + rng.uniform(-0.2, 0.2, len(e)), e, s=16, color=STAGE_COLOR[stage], alpha=0.45, lw=0,
                   zorder=2)
        ax.plot([k - 0.3, k + 0.3], [med, med], color=INK, lw=2.2, solid_capstyle="round", zorder=4)
        ax.text(k + 0.36, med, f"{med:.2f}", va="center", ha="left", fontsize=9.5, color=INK, zorder=5)
        assert e.max() < 6.0, "a point is off the axis"
    ax.set_xticks(range(len(STAGES)), [STAGE_LABEL[s] for s in STAGES])
    ax.set_xlim(-0.6, len(STAGES) - 0.3)
    ax.set_ylim(0, 6.0)
    ax.set_ylabel(f"Error on unseen landmarks\n({UNIT})", fontsize=9.5, color=INK_2)
    ax.set_title("Each stage of the fit, scored on landmarks it never saw", fontsize=11.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=1)
    style(ax)
    ax.tick_params(axis="x", length=0)
    fig.text(0.012, 0.012, "One dot per portrait (100); the black tick and the number are the median.",
             fontsize=8.5, color=INK_3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "stage_error.png", facecolor=SURFACE)
    plt.close(fig)


def region_test(results, out):
    rows = read_csv(results / "region_test.csv")
    med = {(r, s): float(np.median([float(x["error"]) for x in rows if x["region"] == r and x["stage"] == s]))
           for r in problem.REGIONS for s in REGION_STAGES}
    order = sorted(problem.REGIONS, key=lambda r: med[(r, "M0")] - med[(r, "M4")])   # biggest gain on top
    fig, ax = plt.subplots(figsize=(8.0, 3.9), dpi=160)
    for y, region in enumerate(order):
        values = [med[(region, s)] for s in REGION_STAGES]
        ax.plot([min(values), max(values)], [y, y], color=GRID, lw=2.0, zorder=1, solid_capstyle="round")
        for s, label in REGION_STAGES.items():
            ax.scatter(med[(region, s)], y, s=74, color=STAGE_COLOR[s], edgecolors=SURFACE, linewidths=1.6,
                       zorder=3 if s != "M0" else 2, label=label if y == 0 else None)
    ax.set_yticks(range(len(order)), [REGION_LABEL[r] for r in order])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, max(med.values()) * 1.08)
    ax.set_xlabel(f"Median error on the left-out region ({UNIT})", fontsize=9.5, color=INK_2)
    ax.set_title("Predicting a face region the fit was not shown", fontsize=11.5, color=INK, loc="left")
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    style(ax)
    ax.tick_params(axis="y", length=0, labelsize=10, colors=INK)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3, frameon=False, fontsize=9,
              labelcolor=INK_2, handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(out / "region_test.png", facecolor=SURFACE)
    plt.close(fig)


def focal_length(results, out):
    rows = [r for r in read_csv(results / "selection.csv") if r["stage"] == "M3"]
    fm = np.array([float(r["focal_mm"]) for r in rows])
    med, q1, q3 = (np.array([float(r[k]) for r in rows]) for k in ("val_median", "val_q1", "val_q3"))
    eligible = np.array([r["eligible"] == "1" for r in rows])
    chosen = int(np.flatnonzero([r["chosen"] == "1" for r in rows])[0])
    color = STAGE_COLOR["M3"]
    fig, ax = plt.subplots(figsize=(8.0, 4.0), dpi=160)
    ax.fill_between(fm, q1, q3, color=color, alpha=0.12, lw=0, zorder=1)
    ax.plot(fm, med, color=color, lw=2.0, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    ax.scatter(fm[eligible], med[eligible], s=52, color=color, edgecolors=SURFACE, linewidths=1.6, zorder=3)
    ax.scatter(fm[~eligible], med[~eligible], s=44, facecolors=SURFACE, edgecolors=color, linewidths=1.6, zorder=3)
    ax.scatter(fm[chosen], med[chosen], s=150, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=4)
    ax.annotate(f"chosen: {fm[chosen]:g} mm, {med[chosen]:.2f}", (fm[chosen], med[chosen]),
                xytext=(0, -26), textcoords="offset points", ha="center", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(fm, [f"{v:g}" for v in fm])
    ax.minorticks_off()
    lo, hi = q1.min(), q3.max()
    ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.12 * (hi - lo))
    ax.set_xlabel("Focal length shared by all 100 portraits (35 mm equivalent, mm)", fontsize=9.5, color=INK_2)
    ax.set_ylabel(f"Error on validation landmarks\n({UNIT})", fontsize=9.5, color=INK_2)
    ax.set_title("Validation error against the focal length", fontsize=11.5, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    style(ax)
    note = "Line: median over the portraits. Band: quartiles."
    if (~eligible).any():
        note += " Hollow: too many extreme coefficients to be chosen."
    fig.text(0.012, 0.012, note, fontsize=8.5, color=INK_3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "focal_length.png", facecolor=SURFACE)
    plt.close(fig)


def offsets(results, out):
    """Where stage M4 moves each landmark on the mean face, seen from the front."""
    d = np.load(results / "landmark_offsets.npz")
    delta, valid = d["learned_from_all"] * 1000.0, d["valid"]                # mm
    P0, _ = problem.landmark_model(0, 0)
    xy = P0[:, :2] * 1000.0                                                  # mm; camera axes: x right, y down
    size = np.linalg.norm(delta, axis=1)
    stretch = 3.0
    fig, ax = plt.subplots(figsize=(6.4, 6.6), dpi=160)
    ax.scatter(xy[valid, 0], xy[valid, 1], s=5, color="#c9c9cf", lw=0, zorder=1)
    q = ax.quiver(xy[valid, 0], xy[valid, 1], delta[valid, 0] * stretch, delta[valid, 1] * stretch, size[valid],
                  angles="xy", scale_units="xy", scale=1.0, cmap="Blues", clim=(-0.35 * size.max(), size.max()),
                  width=0.0042, headwidth=3.6, headlength=4.2, zorder=2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel("mm", fontsize=9.5, color=INK_2)
    ax.set_ylabel("mm", fontsize=9.5, color=INK_2)
    ax.set_title("Where the detector's landmarks sit, against the first mapping", fontsize=11.5, color=INK,
                 loc="left")
    style(ax)
    bar = fig.colorbar(q, ax=ax, fraction=0.035, pad=0.03)
    bar.ax.set_ylim(0, size.max())
    bar.set_label("offset (mm)", fontsize=9, color=INK_2)
    bar.ax.tick_params(labelsize=8.5, colors=INK_2, length=2)
    bar.outline.set_visible(False)
    fig.text(0.012, 0.012, f"One arrow per landmark ({int(valid.sum())}), drawn {stretch:g} times its length. "
             f"Median {np.median(size[valid]):.2f} mm, largest {size[valid].max():.1f} mm.", fontsize=8.5, color=INK_3)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out / "offsets.png", facecolor=SURFACE)
    plt.close(fig)


def pick_one_per_tone(results):
    """For each prompted tone 1 to 10, the portrait whose final fit error is the median of its tone."""
    fits = np.load(results / "fits_final.npz")
    error = dict(zip(fits["ids"].tolist(), fits["error_pct_iod"].tolist()))
    tone = {r["id"]: int(r["mst_tone_intended"]) for r in read_csv(problem.ROOT / "labels" / "identities_v0.csv")}
    picks = []
    for t in range(1, 11):
        group = sorted((error[i], i) for i in error if tone[i] == t)
        picks.append(group[(len(group) - 1) // 2][1])
    return picks


def overlay(results, out, portraits):
    """Per portrait, three tiles: the portrait, the fitted head's wireframe on it, the head shaded."""
    fits = np.load(results / "fits_final.npz")
    ids = fits["ids"].tolist()
    V0, F, S, E = flame_io.load_flame()
    masks = np.load(problem.FIT / "assets" / "flame_masks.npz")
    eyeball = np.zeros(len(V0), bool)
    eyeball[masks["left_eyeball"]] = eyeball[masks["right_eyeball"]] = True
    is_skin = ~eyeball[F].any(1)
    n_shape, n_expr = fits["shape"].shape[1], fits["expr"].shape[1]
    f, cx, cy = float(fits["focal_px"]), float(fits["cx"]), float(fits["cy"])
    W_full = 2 * cx
    tile, tiles = 300, []
    for pid in pick_one_per_tone(results):
        k = ids.index(pid)
        full = sorted(Path(portraits).glob(f"{pid}.jp*g"))
        img = cv2.imread(str(full[0] if full else problem.ROOT / "images" / "thumbs" / f"{pid}.jpg"))
        scale = img.shape[1] / W_full
        V = V0 + S[:, :, :n_shape] @ fits["shape"][k] + E[:, :, :n_expr] @ fits["expr"][k]
        Q = (V @ flame_io.R_FLIP.T) @ fits["R"][k].T + fits["t"][k]
        uv = np.stack([f * Q[:, 0] / Q[:, 2] + cx, f * Q[:, 1] / Q[:, 2] + cy], axis=1) * scale
        normal = np.cross(Q[F[:, 1]] - Q[F[:, 0]], Q[F[:, 2]] - Q[F[:, 0]])
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        front = normal[:, 2] < 0                                  # triangles that face the camera
        tri = F[front & is_skin]                                  # the wireframe leaves the eyeballs out

        # A square crop around the face landmarks, the same for the three tiles.
        face = uv[masks["face"]]
        centre, half = (face.min(0) + face.max(0)) / 2, 0.62 * (face.max(0) - face.min(0)).max()
        x0, y0 = int(round(centre[0] - half)), int(round(centre[1] - half * 1.08))
        side = int(round(2 * half))

        def crop(image):
            padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_REPLICATE)
            return cv2.resize(padded[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side], (tile, tile),
                              interpolation=cv2.INTER_AREA)

        wire = img.copy()
        edges = np.unique(np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1), axis=0)
        lines = np.round(uv[edges] * 16).astype(np.int32)
        cv2.polylines(wire, list(lines), False, (255, 255, 255), max(1, round(img.shape[1] / 900)), cv2.LINE_AA,
                      shift=4)
        wire = cv2.addWeighted(wire, 0.55, img, 0.45, 0)

        shaded = np.full_like(img, 255)
        light = np.array([0.25, -0.35, -1.0]) / np.linalg.norm([0.25, -0.35, -1.0])
        solid = F[front]
        lam = np.clip(normal[front] @ light, 0.0, 1.0)
        depth = Q[solid].mean(1)[:, 2]
        for j in np.argsort(-depth):                              # far triangles first
            g = int(70 + 170 * lam[j])
            cv2.fillConvexPoly(shaded, np.round(uv[solid[j]] * 16).astype(np.int32), (g, g, g), cv2.LINE_AA,
                               shift=4)
        row = [crop(img), crop(wire), crop(shaded)]
        cv2.putText(row[0], f"{pid}  {fits['error_pct_iod'][k]:.2f} %", (10, tile - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row[0], f"{pid}  {fits['error_pct_iod'][k]:.2f} %", (10, tile - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.concatenate(row, axis=1))
    gap = 12
    rows = []
    for a in range(0, len(tiles), 2):                             # two portraits per row
        rows.append(np.concatenate([tiles[a], np.full((tile, gap, 3), 255, np.uint8), tiles[a + 1]], axis=1))
        rows.append(np.full((gap, rows[-1].shape[1], 3), 255, np.uint8))
    cv2.imwrite(str(out / "overlay.jpg"), np.concatenate(rows[:-1], axis=0), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(problem.RESULTS))
    ap.add_argument("--out", default=str(problem.ROOT / "docs" / "fit"))
    ap.add_argument("--portraits", default=str(problem.ROOT / "portraits" / "identities"))
    a = ap.parse_args()
    results, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stage_error(results, out)
    region_test(results, out)
    focal_length(results, out)
    if (problem.FIT / "assets" / "flame_min.npz").exists():
        offsets(results, out)
        overlay(results, out, a.portraits)
    else:
        print("offsets.png and overlay.jpg skipped: fit/assets/flame_min.npz is not present")
    summary = json.loads((results / "summary.json").read_text())
    print("figures written to", out, "from the", "quick" if summary.get("quick") else "full", "run")


if __name__ == "__main__":
    main()
