"""labels/gallery.json for index.html: one record per portrait, design and measured columns side by side."""
import csv, json
from pathlib import Path
root = Path(__file__).resolve().parent.parent
m = {r["file"][:5]: r for r in csv.DictReader(open(root / "labels/measurements_v1.csv"))}
out = []
for r in csv.DictReader(open(root / "labels/identities_v0.csv")):
    k = m[r["id"]]
    out.append({"id": r["id"], "tone": int(r["mst_tone_intended"]), "age": int(r["age_intended"]), "age_bin": r["age_bin"],
                "gender": r["gender_presentation"], "region": r["region_descriptor"], "skin_words": r["skin_words"],
                "pigment": r["pigment_feature"], "hair": r["hair"], "facial_hair": r["facial_hair"],
                "ita": float(k["ita"]), "delbino": k["delbino"], "L": float(k["L_star"]), "b": float(k["b_star"])})
(root / "labels/gallery.json").write_text(json.dumps(out, indent=0))
print(len(out), "records")
