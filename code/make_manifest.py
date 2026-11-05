"""Identity manifest v0: 100 synthetic adult identities, full factorial
10 Monk Skin Tone bins x 5 age bins x 2 gender presentations = 100 (n=1 per cell).
Region-of-ancestry and pigment features are PROMPT DESCRIPTORS for feature diversity,
not labels. Labels come from measurement after generation (ITA, estimated age/gender,
embedding de-duplication). Deterministic: seed 20260918."""
import csv, random
random.seed(20260918)

MST = list(range(1, 11))  # Monk Skin Tone 1 (lightest) .. 10 (deepest)
AGE_BINS = [("18-29", 18, 29), ("30-44", 30, 44), ("45-59", 45, 59), ("60-74", 60, 74), ("75+", 75, 88)]
GENDERS = ["woman", "man"]

# Region descriptors weighted per tone; overlapping on purpose so tone and region are not confounded.
REGIONS = {
    1: ["Northern European", "Eastern European", "East Asian", "mixed heritage"],
    2: ["Northern European", "Southern European", "East Asian", "mixed heritage"],
    3: ["Southern European", "East Asian", "Southeast Asian", "Middle Eastern", "Latin American", "mixed heritage"],
    4: ["Southern European", "Middle Eastern", "Southeast Asian", "Latin American", "Central Asian", "mixed heritage"],
    5: ["Middle Eastern", "South Asian", "Southeast Asian", "Latin American", "North African", "mixed heritage"],
    6: ["South Asian", "Southeast Asian", "Latin American", "North African", "Pacific Islander", "mixed heritage"],
    7: ["South Asian", "Sub-Saharan African", "Latin American", "Pacific Islander", "mixed heritage"],
    8: ["Sub-Saharan African", "South Asian", "Aboriginal Australian", "Afro-Caribbean", "mixed heritage"],
    9: ["Sub-Saharan African", "Afro-Caribbean", "South Asian", "Melanesian", "mixed heritage"],
    10: ["Sub-Saharan African", "Afro-Caribbean", "Melanesian", "mixed heritage"],
}
MST_WORDS = {
    1: "very light, pale skin", 2: "light skin", 3: "light skin with warm undertone", 4: "light-medium skin",
    5: "medium skin", 6: "medium-tan skin", 7: "tan-brown skin", 8: "brown skin", 9: "dark brown skin", 10: "very dark, deep brown skin",
}
HAIR = ["short cropped hair", "hair tied back", "shaved head", "short hair", "hair pulled back tightly"]

def pigment_feature(tone, lo, gender):
    """About 20 of 100 get a visible pigment pattern; the rest 'none'. Chosen once per row."""
    r = random.random()
    if r > 0.22:
        return "none"
    opts = ["scattered moles"]
    if tone <= 4: opts += ["freckles across the nose and cheeks"]
    if lo >= 45: opts += ["a few solar lentigines on the cheeks and temples"]
    if 30 <= lo <= 59 and gender == "woman": opts += ["mild melasma on the cheeks"]
    if tone >= 6: opts += ["a few post-inflammatory dark marks on the cheeks"]
    if random.random() < 0.08: opts += ["small vitiligo patches near the mouth"]
    return random.choice(opts)

rows = []
i = 0
for tone in MST:
    for (label, lo, hi) in AGE_BINS:
        for gender in GENDERS:
            i += 1
            age = random.randint(lo, hi)
            region = random.choice(REGIONS[tone])
            hair = random.choice(HAIR)
            facial_hair = "none"
            if gender == "man" and random.random() < 0.35:
                facial_hair = random.choice(["light stubble", "short trimmed beard", "moustache"])
            rows.append({
                "id": f"ID{i:03d}",
                "mst_tone_intended": tone,
                "skin_words": MST_WORDS[tone],
                "age_bin": label,
                "age_intended": age,
                "gender_presentation": gender,
                "region_descriptor": region,
                "pigment_feature": pigment_feature(tone, lo, gender),
                "hair": hair,
                "facial_hair": facial_hair,
                "seed": 20260918 + i,
                # measured after generation (leave empty now)
                "ita_measured": "", "mst_measured": "", "age_estimated": "", "gender_estimated": "",
                "dup_check": "", "status": "planned",
            })

# v0.1 (18 Sep, after ID005 came out bald-looking): no "shaved head" on women's rows not yet generated.
# Applied after the draws so the random sequence, and every other column, is unchanged.
for r in rows:
    if r["gender_presentation"] == "woman" and r["hair"] == "shaved head" and r["id"] > "ID005":
        r["hair"] = "hair tied back"

with open("labels/design_v0.csv", "w", newline="") as f:   # the unmeasured design; labels/identities_v0.csv is this sheet as used (same rows, status set, empty measurement columns dropped)
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(len(rows), "rows;", sum(r["pigment_feature"] != "none" for r in rows), "with a pigment feature")
