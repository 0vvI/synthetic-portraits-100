"""MediaPipe Face Landmarker on the 100 portraits -> fit/results/landmarks_mp478.npz (SPEC.md 7).

    python fit/landmarks.py [--portraits portraits/identities] [--model fit/assets/face_landmarker.task]

The model file (Apache-2.0, 3.7 MB) is not in this repository. Download it once:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--portraits", default=str(ROOT / "portraits" / "identities"))
    ap.add_argument("--model", default=str(ROOT / "fit" / "assets" / "face_landmarker.task"))
    ap.add_argument("--out", default=str(ROOT / "fit" / "results" / "landmarks_mp478.npz"))
    a = ap.parse_args()

    opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=a.model),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
    )
    files = sorted(Path(a.portraits).glob("ID*.jp*g"))
    ids, xy, z, size = [], [], [], []
    with vision.FaceLandmarker.create_from_options(opts) as detector:
        for path in files:
            bgr = cv2.imread(str(path))
            h, w = bgr.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if len(res.face_landmarks) != 1 or len(res.face_landmarks[0]) != 478:
                raise SystemExit(f"{path.name}: expected one face with 478 landmarks")
            lm = res.face_landmarks[0]
            ids.append(path.stem)
            xy.append([[p.x * w, p.y * h] for p in lm])
            z.append([p.z * w for p in lm])
            size.append([w, h])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        ids=np.array(ids),
        xy=np.array(xy, dtype=np.float32),
        z=np.array(z, dtype=np.float32),
        size=np.array(size, dtype=np.int32),
    )
    print(f"wrote {out}: {len(ids)} portraits, xy {np.array(xy).shape}")


if __name__ == "__main__":
    main()
