"""Native-resolution cheek close-ups, one per intended tone.  python3 code/skin_strip.py PORTRAITS_DIR
For each tone, the portrait whose measured ITA is closest to that tone's median; the crop is centred on the
same landmark-placed cheek patch measure.py uses, 320 px of the 1792 x 2400 original, shown 1:1."""
import csv, sys, statistics as st
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src = Path(sys.argv[1]); root = Path(__file__).resolve().parent.parent
ids = list(csv.DictReader(open(root / "labels/identities_v0.csv")))
ita = {r["file"][:5]: float(r["ita"]) for r in csv.DictReader(open(root / "labels/measurements_v1.csv"))}
det = cv2.FaceDetectorYN.create(str(root / "code/face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 5000)
C, pad, top = 320, 8, 46
fig = Image.new("RGB", (10 * (C + pad) + pad, C + top + pad), (24, 24, 26))
d = ImageDraw.Draw(fig)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
except OSError:
    f = ImageFont.load_default()
for t in range(1, 11):
    rows = [r for r in ids if int(r["mst_tone_intended"]) == t]
    med = st.median(ita[r["id"]] for r in rows)
    r = min(rows, key=lambda r: abs(ita[r["id"]] - med))
    img = cv2.imread(str(src / f"{r['id']}.jpeg")); h, w = img.shape[:2]
    for sc in (640.0 / w, 960.0 / w, 1.0):   # 640 first, as measure.py v1
        im2 = img if sc == 1.0 else cv2.resize(img, (int(w * sc), int(h * sc)))
        det.setInputSize((im2.shape[1], im2.shape[0])); _, faces = det.detect(im2)
        if faces is not None and len(faces):
            b = max(faces, key=lambda q: q[2] * q[3]) / sc; break
    fw = b[2]; (rex, rey), _, _, (rmx, rmy), _ = b[4:14].reshape(5, 2)
    cx, cy = int((rex + rmx) / 2 - fw * 0.08), int((rey + rmy) / 2)
    crop = img[cy - C // 2:cy + C // 2, cx - C // 2:cx + C // 2][:, :, ::-1]
    x = pad + (t - 1) * (C + pad)
    fig.paste(Image.fromarray(np.ascontiguousarray(crop)), (x, top))
    d.text((x + 2, 6), f"tone {t}  {r['id']}", fill=(225, 225, 228), font=f)
    d.text((x + 2, 25), f"ITA {ita[r['id']]:.1f}°", fill=(160, 160, 168), font=f)
fig.save(root / "docs/skin_detail.jpg", quality=90)
print("wrote docs/skin_detail.jpg")
