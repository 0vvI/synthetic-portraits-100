"""Flatten each manifest row into ONE line for the Gemini composer. Usage:
python3 build_prompts.py identities_v0.csv ID001            -> prints the prompt
python3 build_prompts.py identities_v0.csv --all > prompts_identities_v0.txt
Each line is tagged with a unique marker [HDS-<id>] so a chat can be found again by text."""
import csv, sys
path, sel = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(path)))
def prompt(r):
    pig = "" if r["pigment_feature"] == "none" else f" Natural skin with {r['pigment_feature']}."
    fh = "" if r["facial_hair"] == "none" else f" {r['facial_hair']},"
    who = f"{r['age_intended']}-year-old {r['gender_presentation']}"
    return (f"[HDS-{r['id']}] Photorealistic studio portrait photograph of a {who} of {r['region_descriptor']} heritage "
            f"with {r['skin_words']}.{pig} {r['hair']},{fh} forehead and both ears visible, no glasses, no jewellery, "
            f"no makeup, neutral relaxed expression, mouth closed, eyes open looking at the camera. Head and shoulders, "
            f"face centred and front-facing, camera at eye level, 85 mm lens. Soft even diffuse lighting from the front, "
            f"no strong shadows, no specular hot spots, plain mid-grey background. Unretouched natural skin: pores, fine "
            f"lines and natural colour variation visible. No skin smoothing.")
if sel == "--all":
    for r in rows: print(r["id"] + "\t" + prompt(r))
else:
    print(next(prompt(r) for r in rows if r["id"] == sel))
