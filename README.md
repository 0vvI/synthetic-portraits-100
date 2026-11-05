# Synthetic Portraits 100

One hundred AI-generated studio portraits of people who do not exist, one for each combination of **10 prompted
skin tones × 5 age bins × 2 gender presentations**, with skin tone measured from every image. The design balances
skin appearance rather than race: the same number of portraits for every prompted tone. The measurements then
check whether the generator actually delivered the tones it was asked for: at the ends of the scale it did, but
the middle tones came out almost the same.

**[Browse the gallery](https://0vvi.github.io/synthetic-portraits-100/)** (filter by tone, age and gender) ·
[What the measurements show](#what-the-measurements-show) · [3D skin materials](#3d-skin-materials) ·
[Reproduce](#reproduce)

![contact sheet](docs/contact_sheet.jpg)

*All 100, one row per prompted Monk Skin Tone (1 = lightest, 10 = deepest); along a row, the five age bins
from 18–29 to 75+, a woman and a man in each.*

## Highlights

- **A balanced design, delivered exactly:** 10 per tone, 20 per age bin, 50 women and 50 men.
- **Skin tone measured, not assumed:** face detection, two landmark-placed cheek patches, CIELAB colour and the
  Individual Typology Angle (ITA) for every image.
- **A clear finding:** the prompt phrases for tones 2–6 barely change the measured skin tone, and the set skews light even
  though every prompted tone has the same count.
- **Reproducible and traceable:** the design sheet and all 100 prompts regenerate exactly from one seed, and every
  original image has a recorded sha256.
- **3D skin materials:** seven identities' skin textures rendered on one head as a tone ladder, plus full heads
  for ID068 and ID089, with each ladder skin's front-render tone measured against its texture and its portrait.

## What the measurements show

![measured ITA by prompted tone](docs/ita_by_tone.png)

| Prompted tone | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| median ITA (°) | 65.50 | 55.35 | 53.25 | 51.50 | 52.00 | 51.40 | 44.40 | 36.55 | 25.40 | 4.95 |

- The measured cheek ITA falls with the prompted tone overall, but **tones 2–6 overlap almost completely**:
  the prompt phrases for tones 2–6, from "light skin" to "medium-tan skin", barely move the measured cheek ITA.
  From tone 7 to 10 the medians fall steadily, with wide overlap between neighbours.
- Measured, the set skews light: 26 very light, 45 light, 10 intermediate, 13 tan, 5 brown, 1 dark. Equal
  counts per prompted tone do not give equal coverage of measured tone.
- The obvious next experiment is to steer the middle tones with something more concrete than a phrase, such as
  a colour reference.

![cheek close-ups](docs/skin_detail.jpg)

*Unretouched skin at the original resolution (320 px crops of 1792 × 2400 images): for each prompted tone,
the portrait whose measured ITA is closest to that tone's median.*

## What is in it

| Path | Contents |
|---|---|
| `images/thumbs/ID001.jpg` … `ID100.jpg` | the portraits at 600 × 800 (the originals are 1792 × 2400 JPEG) |
| `labels/identities_v0.csv` | the design sheet: prompted tone, age, gender presentation and prompt descriptors |
| `labels/measurements_v1.csv` | per image: face box, cheek ITA, the two cheeks' ITA difference, L\*, b\*, Del Bino class, sharpness, duplicate hash |
| `labels/provenance.json` | per image: the generator's file name, sha256 and size |
| `prompts/` | the prompt template and the 100 prompts exactly as sent |
| `code/` | design sheet, prompts, harvest, measurement, figures and gallery data |
| `index.html` | the browsable gallery, served at the link above |
| `labels/texture_ita.csv` | for the 31 identities with a gated skin texture: the texture's and the portrait's cheek ITA |
| `docs/3d/` | 3D renders: the ID012 hero, the seven-skin tone ladder, full heads ID068 and ID089, two tone charts |

## How it was made

1. **Design sheet** (`code/make_manifest.py`, seed 20260918). Ten Monk Skin Tone bins × five adult age bins
   (18–29, 30–44, 45–59, 60–74, 75+) × two gender presentations, one identity per cell. Each row also gets
   a region-of-ancestry descriptor, a hair style, optional facial hair and, for 30 rows, a pigment feature
   (moles, freckles, solar lentigines, post-inflammatory marks, mild melasma). The region lists of
   neighbouring tones overlap, which reduces but does not remove the link between region and tone. The
   descriptors steer the prompt; they are not labels. After the first five portraits, "shaved head" was
   dropped from women's rows (v0.1, 14 rows); the repository copy of the script applies that rule, so it
   reproduces the sheet that was used.
2. **One prompt per row** (`code/build_prompts.py`, `prompts/PROMPT_TEMPLATE.md`): a front-facing,
   evenly lit, unretouched head-and-shoulders portrait on mid-grey, forehead and ears visible, no glasses,
   jewellery or make-up. The generator takes no seed; the `seed` column is the row's own.
3. **Generation:** Gemini web app, Images mode (Nano Banana 2), aspect 3:4, full-size download
   (1792 × 2400), 18–20 Sep 2026. Each download was copied once with its sha256 (`code/harvest.py`) and
   never edited. Two portraits were regenerated (ID001, first made at the wrong aspect ratio; ID092, too
   close to ID082) and one prompt was re-sent after a lost render (ID058).
4. **Measurement** (`code/measure.py`): YuNet face detection at 640 px width (one face in every image),
   two landmark-placed cheek patches, the median CIELAB colour of each, ITA = atan((L\* − 50) / b\*) in
   degrees for each cheek, averaged over the two; the Del Bino class follows from ITA. Near-duplicates are
   screened with a 64-bit DCT hash of the face: four pairs fall under the threshold, and all four were
   checked by eye and are different people.

Counts came out exactly as designed: 10 per tone, 20 per age bin, 50 women and 50 men.

## 3D skin materials

![seven identities' skin on one 3D head](docs/3d/tone_ladder.jpg)

*Seven identities' skin on one head, ordered by the texture's measured cheek ITA (+69° to −55°): ID004, ID012,
ID056, ID074, ID068, ID077, ID089. The head, eyes, brow shape and lights are the same for every skin, and every shot
uses the hero's exposure, so differences in brightness are the skin's. Only the identity's skin maps change:
colour, subdermal colour, scattering weight and specular, with the ear's colour tile matched to each skin.*

Each skin texture is made from the identity's text description, not from its portrait: an image model recolours
a template UV skin texture toward the description, and automatic gates check every result at 2048, 4096 and
8192 px (layout, openings, lips, nose, baked shading, fine detail). 31 of the 100 identities have a texture that
passes every gate, covering prompted tones 1–9 (none at tone 10). A private head pipeline then adds displacement,
eyes, lashes, brows and shading maps, and Blender Cycles renders the head with random-walk subsurface scattering
from 4K material maps. The head shape is one generic FLAME 2023 Open head for every skin, not a fit to the
portrait, so a render shows the identity's skin, not a 3D likeness of the photograph. The texture and head
pipeline is not part of this repository.

### The hero: ID012

![ID012 turntable](docs/3d/ID012_turntable.gif)

![ID012 in four lighting setups](docs/3d/ID012_renders.jpg)

![ID012 close-ups](docs/3d/ID012_closeups.jpg)

*ID012 (prompted tone 2: a 27-year-old man of East Asian heritage, "light skin", shaved head): front,
three-quarter, Rembrandt and backlit light, then cheek, eye and lip close-ups.*

### Two full heads at darker tones: ID068 and ID089

![ID068 in four lighting setups](docs/3d/ID068_renders.jpg)

![ID089 in four lighting setups](docs/3d/ID089_renders.jpg)

*ID068 (prompted tone 7: a 68-year-old man of Sub-Saharan African heritage, "tan-brown skin") and ID089 (prompted
tone 9: an 80-year-old woman of Afro-Caribbean heritage, "dark brown skin").*

In the ladder only four skin maps (and the ear's colour tile) change per identity. These two went through the whole pipeline on their own
texture, so their surface maps are their own as well: colour, subdermal colour, scattering weight, specular,
micro height and normal, roughness, and the ear tiles. The scattering-radius and micro-detail-weight maps are the
same generic maps on every head. Shared with the hero: the head shape, the eye and brow method (the brow shape is
a stock mask, not the identity's), the lights, and the hero's exposure for each shot.

### Does the rendered skin keep the tone?

![texture and portrait cheek ITA against prompted tone](docs/3d/texture_vs_portrait_ita.png)

- On the 31 gated textures, the texture's cheek ITA follows the prompted tone more closely than the portrait's:
  Spearman ρ −0.96, against −0.83 for the portraits (`labels/texture_ita.csv`). The texture is an unlit colour
  map and the portrait a studio-lit photograph, so compare their orderings, not their levels: the darkest
  textures are far darker than their portraits.

![render cheek ITA against texture and portrait](docs/3d/render_vs_texture_ita.png)

- Each ladder skin's front render is measured the same way as the portraits (YuNet at 640 px, two cheek patches;
  `docs/3d/render_vs_texture_ita.csv`). The renders keep the textures' order exactly (Spearman ρ 1.0 over the
  seven) but read lighter than their textures: by +5° to +8° for the four lightest (ID004 +5.2, ID056 +5.3,
  ID012 +8.0, ID074 +8.0) and by more for the three darkest (ID068 +11.8, ID077 +15.1, ID089 +26.9). The renders
  are not tone-accurate, least of all at the dark end.
- Against the portraits the order differs (Spearman ρ 0.89): ID068 and ID077 render darker than ID074 although
  their portraits are lighter, because a render's tone comes from its texture.

## Limits

- One identity per cell: good for coverage, too small for per-cell statistics.
- ITA comes from studio-lit JPEGs, not a colour-calibrated capture. On 29 portraits the two cheeks differ by
  more than 20° (side light, stubble or shadow); the `ita_lr_diff` column flags them.
- The people are synthetic, but a generated face can resemble a real person by chance.
- Estimated age and gender are not measured yet.
- Adults only (18+).
- 3D: every skin is on one generic head shape, so no render is a likeness of its portrait; 7 of the 31
  textures are shown rendered, and the renders read lighter than their textures, most at the dark end.

## Reproduce

Run from the repository root (Python 3, `numpy`, `opencv-python` 5.0, `Pillow`, `matplotlib`):

```
python3 code/make_manifest.py                                  # labels/design_v0.csv from seed 20260918
python3 code/build_prompts.py labels/identities_v0.csv --all > prompts/prompts_identities_v0.txt
python3 code/measure.py identities                             # labels/measurements_v1.csv from portraits/identities/
python3 code/make_sheets.py portraits/identities               # images/thumbs/ and docs/contact_sheet.jpg
python3 code/skin_strip.py portraits/identities                # docs/skin_detail.jpg
python3 code/plot_ita.py portraits/identities                  # docs/ita_by_tone.png
python3 code/make_gallery_data.py                              # labels/gallery.json for index.html
python3 code/make_social_card.py                               # docs/social_card.jpg, the gallery's link preview
```

`make_manifest.py` writes the unmeasured design sheet; its design columns match `labels/identities_v0.csv`
row for row, and `build_prompts.py` reproduces the 100 prompts byte for byte (both checked 5 Nov 2026).
Re-running `measure.py` on the originals reproduces `labels/measurements_v1.csv` (checked 4 Nov 2026). The
full-size images (287 MB) are not in this repository; put them in `portraits/identities/`.

## Terms and use

The portraits were made with Google Gemini. Google states that it does not claim ownership of generated
content. Its terms also list "using AI-generated content from our services to develop machine learning
models or related AI technology" as abuse, so this set is published for viewing and evaluation, not as
training data. See Google's [Terms of Service](https://policies.google.com/terms) and
[Generative AI Prohibited Use Policy](https://policies.google.com/terms/generative-ai/use-policy).
Google says Gemini images carry an invisible SynthID watermark; the originals are byte-identical to the
downloads.

## Who wrote what

Design, prompts, generation runs and measurement: Claude Code sessions directed by Ao Zuo, 18–20 Sep 2026.
This repository, the measurement fix and the public write-up: the same way, 4–5 Nov 2026.

## Licence

- Code (`code/`, `index.html`): MIT, see `LICENSE`.
- Images, labels and prompts: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), see
  `LICENSE-DATA`. Credit: "Synthetic Portraits 100, Ao Zuo, 2026".
- `code/face_detection_yunet_2023mar.onnx` is from [opencv_zoo](https://github.com/opencv/opencv_zoo),
  MIT licence, copyright (c) 2020 Shiqi Yu (`code/YUNET_LICENSE`).
- The 3D renders use the FLAME 2023 Open head model (Li et al., Max Planck Institute for Intelligent Systems),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
