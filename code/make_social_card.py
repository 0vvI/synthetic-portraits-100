"""docs/social_card.jpg: the 1200 x 630 link-preview image for index.html, from images/thumbs/.
24 portraits evenly spaced through the ID order (tone-major), so the mosaic runs from tone 1 to tone 10.
Run from the repository root: python3 code/make_social_card.py"""
from pathlib import Path
from PIL import Image

root = Path(".")
cols, rows, w, h, top = 8, 3, 150, 200, 15
ids = [f"ID{1 + round(i * 99 / (cols * rows - 1)):03d}" for i in range(cols * rows)]
card = Image.new("RGB", (cols * w, 630), (24, 24, 27))
for i, pid in enumerate(ids):
    im = Image.open(root / "images/thumbs" / f"{pid}.jpg").convert("RGB").resize((w, h), Image.LANCZOS)
    card.paste(im, ((i % cols) * w, top + (i // cols) * h))
card.save(root / "docs/social_card.jpg", quality=88, optimize=True)
print("docs/social_card.jpg from", ", ".join(ids))
