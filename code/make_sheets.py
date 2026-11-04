"""Repo-sized copies and the contact sheet.  python3 code/make_sheets.py PORTRAITS_DIR
Writes images/thumbs/IDxxx.jpg (600 x 800) and docs/contact_sheet.jpg (10 x 10, rows = intended Monk tone 1-10)."""
import csv, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

src = Path(sys.argv[1]); root = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(root / "labels/identities_v0.csv")))
(root / "images/thumbs").mkdir(parents=True, exist_ok=True)
W, H, pad, lab = 240, 320, 6, 22
sheet = Image.new("RGB", (10 * (W + pad) + pad, 10 * (H + lab + pad) + pad), (24, 24, 26))
d = ImageDraw.Draw(sheet)
try:
    f = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 15)
except OSError:
    f = ImageFont.load_default()
rows.sort(key=lambda r: (int(r["mst_tone_intended"]), r["id"]))
for k, r in enumerate(rows):
    im = Image.open(src / f"{r['id']}.jpeg").convert("RGB")
    im.resize((600, 800), Image.LANCZOS).save(root / f"images/thumbs/{r['id']}.jpg", quality=86)
    x, y = pad + (k % 10) * (W + pad), pad + (k // 10) * (H + lab + pad)
    sheet.paste(im.resize((W, H), Image.LANCZOS), (x, y))
    d.text((x + 3, y + H + 3), f"{r['id']}  MST {r['mst_tone_intended']}  {r['age_intended']}", fill=(200, 200, 200), font=f)
sheet.save(root / "docs/contact_sheet.jpg", quality=88)
print("thumbs and sheet written")
