"""Write fit/results/regions_mp468.json: the landmark ids of each face region, taken from MediaPipe's own
connection tables, so the evaluation does not need MediaPipe installed.

    python fit/regions.py
"""
import json
from pathlib import Path

from mediapipe.tasks.python import vision

OUT = Path(__file__).resolve().parent / "results" / "regions_mp468.json"
TABLES = {
    "face_oval": ["FACE_LANDMARKS_FACE_OVAL"],
    "lips": ["FACE_LANDMARKS_LIPS"],
    "nose": ["FACE_LANDMARKS_NOSE"],
    "eyes": ["FACE_LANDMARKS_LEFT_EYE", "FACE_LANDMARKS_RIGHT_EYE"],
    "brows": ["FACE_LANDMARKS_LEFT_EYEBROW", "FACE_LANDMARKS_RIGHT_EYEBROW"],
    "left_eye": ["FACE_LANDMARKS_LEFT_EYE"],
    "right_eye": ["FACE_LANDMARKS_RIGHT_EYE"],
}


def main():
    conns = vision.FaceLandmarksConnections
    regions = {}
    for name, tables in TABLES.items():
        ids = {i for t in tables for c in getattr(conns, t) for i in (c.start, c.end)}
        regions[name] = sorted(ids)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(regions, indent=1) + "\n")
    print({k: len(v) for k, v in regions.items()})


if __name__ == "__main__":
    main()
