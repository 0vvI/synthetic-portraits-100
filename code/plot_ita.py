"""Measured cheek ITA against intended Monk Skin Tone.  python3 code/plot_ita.py PORTRAITS_DIR
One dot per portrait, filled with its own measured cheek colour; a black tick marks each tone's median.
Grey bands are the Del Bino classes the ITA thresholds define."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

src = Path(sys.argv[1]); root = Path(__file__).resolve().parent.parent
ids = {r["id"]: r for r in csv.DictReader(open(root / "labels/identities_v0.csv"))}
meas = {r["file"][:5]: r for r in csv.DictReader(open(root / "labels/measurements_v1.csv"))}
rng = np.random.default_rng(1)
# the same landmark-placed cheek patches as measure.py, so each dot's colour is the pixels its ITA came from
import cv2
yunet = Path(__file__).resolve().parent / "face_detection_yunet_2023mar.onnx"   # opencv_zoo, MIT
det = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.6, 0.3, 5000)
def cheek_rgb(path):
    img = cv2.imread(str(path)); h, w = img.shape[:2]
    for sc in (640.0 / w, 960.0 / w, 1.0):   # 640 first, as measure.py v1
        im2 = img if sc == 1.0 else cv2.resize(img, (int(w * sc), int(h * sc)))
        det.setInputSize((im2.shape[1], im2.shape[0])); _, faces = det.detect(im2)
        if faces is not None and len(faces):
            b = max(faces, key=lambda r: r[2] * r[3]) / sc; break
    x, y, fw, fh = b[:4]; (rex, rey), (lex, ley), _, (rmx, rmy), (lmx, lmy) = b[4:14].reshape(5, 2)
    pw = int(fw * 0.14)
    cen = [((rex + rmx) / 2 - fw * 0.08, (rey + rmy) / 2), ((lex + lmx) / 2 + fw * 0.08, (ley + lmy) / 2)]
    px = [img[int(cy - pw / 2):int(cy + pw / 2), int(cx - pw / 2):int(cx + pw / 2)].reshape(-1, 3) for cx, cy in cen]
    return np.median(np.concatenate(px), 0)[::-1] / 255.0
xs, ys, cs = [], [], []
for i, r in ids.items():
    xs.append(int(r["mst_tone_intended"]) + rng.uniform(-0.18, 0.18)); ys.append(float(meas[i]["ita"]))
    cs.append(cheek_rgb(src / f"{i}.jpeg"))
xs, ys = np.array(xs), np.array(ys)

fig, ax = plt.subplots(figsize=(9, 5.8), dpi=160)
bands = [(55, 90, "very light"), (41, 55, "light"), (28, 41, "intermediate"), (10, 28, "tan"), (-30, 10, "brown"), (-90, -30, "dark")]
for k, (lo, hi, name) in enumerate(bands):
    if k % 2 == 0:
        ax.axhspan(lo, hi, color="#f1f1f3", zorder=0, lw=0)
    ax.text(10.62, (max(lo, -62) + min(hi, 85)) / 2, name, va="center", ha="left", fontsize=8.5, color="#6b6b75")
ax.scatter(xs, ys, s=58, c=cs, edgecolors="#2b2b30", linewidths=0.8, zorder=3)
for t in range(1, 11):
    med = np.median(ys[np.round(xs) == t])
    ax.plot([t - 0.32, t + 0.32], [med, med], color="#111114", lw=2.2, solid_capstyle="round", zorder=4)
ax.set_xlim(0.4, 10.6); ax.set_ylim(-62, 85)   # every portrait is on the plot (min ITA -53.9)
ax.set_xticks(range(1, 11))
ax.set_xlabel("Intended Monk Skin Tone (prompt)", fontsize=10, color="#2b2b30")
ax.set_ylabel("Measured cheek ITA (°)", fontsize=10, color="#2b2b30")
ax.set_title("Prompted tone vs measured skin lightness, 100 portraits", fontsize=11.5, color="#111114", loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#9a9aa3")
ax.tick_params(colors="#4a4a52", labelsize=9)
ax.grid(axis="y", color="#e4e4e8", lw=0.6, zorder=1)
fig.tight_layout()
fig.savefig(root / "docs/ita_by_tone.png", facecolor="white")
assert ys.min() > -62 and ys.max() < 85, "a point is off the axis"
print("medians", [round(float(np.median(ys[np.round(xs) == t])), 1) for t in range(1, 11)])
