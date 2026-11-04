"""Measure accepted portraits: face box, cheek ITA (CIELAB), Del Bino skin class, sharpness, and a DCT hash for
near-duplicate detection. Writes labels/measurements_v1.csv (this script is the only writer of that file).
Usage: python3 measure.py identities
Age/gender estimation (FairFace classifier) is a later step; it needs torchvision + the public weights."""
import sys, os, glob, csv, math
import numpy as np, cv2
setname = sys.argv[1]
files = sorted(glob.glob(f"portraits/{setname}/*.jpeg") + glob.glob(f"portraits/{setname}/*.jpg") + glob.glob(f"portraits/{setname}/*.png"))
YUNET = "code/face_detection_yunet_2023mar.onnx"   # optional; from github.com/opencv/opencv_zoo (MIT, Shiqi Yu)
det = cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.6, 0.3, 5000) if os.path.exists(YUNET) and hasattr(cv2, "FaceDetectorYN") else None

def detect(img):
    """Largest face box (x,y,w,h) via YuNet if the model is present; else a fixed heuristic for a centred
    head-and-shoulders portrait (face ~ 45% of width, centred, upper-middle). Returns (box, method)."""
    h, w = img.shape[:2]
    if det is not None:
        # v1 (4 Nov): 640 px first. At full size (1792 px) YuNet returned a partial face box on 48 of 100
        # portraits, which put the cheek patches beside the nose or under the eye.
        for scale in (640.0 / w, 960.0 / w, 1.0):
            im2 = img if scale == 1.0 else cv2.resize(img, (int(w * scale), int(h * scale)))
            det.setInputSize((im2.shape[1], im2.shape[0])); _, faces = det.detect(im2)
            if faces is not None and len(faces):
                best = max(faces, key=lambda r: r[2] * r[3]) / scale
                x, y, fw, fh = best[:4]; lm = best[4:14].reshape(5, 2)
                return (int(x), int(y), int(fw), int(fh)), ("yunet" if scale == 1.0 else f"yunet@{int(w*scale)}"), lm, len(faces)
    fw = int(w * 0.45); fh = int(fw * 1.3)
    return (int(w / 2 - fw / 2), int(h * 0.42 - fh / 2), fw, fh), "heuristic", None, 0

def ita_of(patch_bgr):
    lab = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    L = lab[:, 0] * 100.0 / 255.0; b = lab[:, 2] - 128.0
    Lm, bm = np.median(L), np.median(b)
    return math.degrees(math.atan2(Lm - 50.0, bm if abs(bm) > 1e-6 else 1e-6)), Lm, bm

def delbino(ita):
    return ("very light" if ita > 55 else "light" if ita > 41 else "intermediate" if ita > 28
            else "tan" if ita > 10 else "brown" if ita > -30 else "dark")

def dct_hash(gray):
    g = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8]; med = np.median(d[1:, 1:])
    return "".join("1" if v > med else "0" for v in d.flatten())

rows = []
for f in files:
    img = cv2.imread(f); h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (x, y, fw, fh), method, lm, nfaces = detect(img)
    pw = int(fw * 0.14)
    if lm is not None:
        # cheek centre per side: between the eye and the mouth corner, pushed 8% of face width outward (avoids nose shadow, beard, ears)
        (rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy) = lm
        rcx, rcy = (rex + rmx) / 2 - fw * 0.08, (rey + rmy) / 2
        lcx, lcy = (lex + lmx) / 2 + fw * 0.08, (ley + lmy) / 2
    else:
        rcx = x + fw * 0.22; lcx = x + fw * 0.78; rcy = lcy = y + fh * 0.58
    def patch(cx_, cy_): return img[int(cy_ - pw / 2):int(cy_ + pw / 2), int(cx_ - pw / 2):int(cx_ + pw / 2)]
    lp, rp = patch(rcx, rcy), patch(lcx, lcy)
    ita_l, L_l, b_l = ita_of(lp); ita_r, L_r, b_r = ita_of(rp); ita = (ita_l + ita_r) / 2
    sharp = cv2.Laplacian(gray[y:y + fh, x:x + fw], cv2.CV_64F).var()
    centred = abs((x + fw / 2) / w - 0.5) < 0.12
    rows.append({"file": os.path.basename(f), "face": f"{x},{y},{fw},{fh}", "face_method": method, "n_faces": nfaces, "face_frac": round(fw / w, 3),
                 "centred": centred, "ita": round(ita, 1), "ita_lr_diff": round(abs(ita_l - ita_r), 1),
                 "L_star": round((L_l + L_r) / 2, 1), "b_star": round((b_l + b_r) / 2, 1),
                 "delbino": delbino(ita), "sharpness": round(sharp, 1), "dct_hash": dct_hash(gray[y:y + fh, x:x + fw])})

# near-duplicates: Hamming distance on the 64-bit face hash
for i, r in enumerate(rows):
    r["near_dups"] = ""
    if "dct_hash" not in r: continue
    d = [(sum(a != b for a, b in zip(r["dct_hash"], s["dct_hash"])), s["file"]) for j, s in enumerate(rows) if j != i and "dct_hash" in s]
    r["near_dups"] = ";".join(f"{fn}({dist})" for dist, fn in sorted(d) if dist <= 5)

out = "labels/measurements_v1.csv" if setname == "identities" else f"labels/measurements_{setname}_v0.csv"
keys = ["file", "face", "face_method", "n_faces", "face_frac", "centred", "ita", "ita_lr_diff", "L_star", "b_star", "delbino", "sharpness", "dct_hash", "near_dups"]
with open(out, "w", newline="") as fh_:
    w = csv.DictWriter(fh_, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(len(rows), "images measured ->", out)
