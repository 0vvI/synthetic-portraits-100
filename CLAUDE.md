# Notes for Claude sessions working on this repository

This is Ao Zuo's portfolio project. Every number in it must be correct. Two kinds of session work on it:

- **The local session** on Ao's Mac. It has the full-size portraits, the private texture and head pipeline,
  Blender, and the 31 skin textures. It produces everything under `docs/3d/`, plus `labels/texture_ita.csv`.
- **Cloud sessions** (this file is mostly for you). You have only this repository. You own the write-up and the
  site: `README.md`, `index.html`, `code/`, `LICENSE`, `LICENSE-DATA`.

## Who edits what

| Path | Owner | Cloud sessions |
|---|---|---|
| `docs/3d/**`, `labels/texture_ita.csv`, `CLAUDE.md` | local session | read, link, describe; never edit or regenerate |
| `fit/results/**`, `fit/embedding/*.npz` and `.csv`, `docs/fit/**` | local session (needs the FLAME model and the full-size portraits) | read, link, describe; never edit or regenerate |
| `fit/**` code, `fit/SPEC.md`, `fit/README.md`, `.github/workflows/fitcore.yml` | local session | propose changes in a pull request; the tests (`ctest`, `pytest fit/tests`) must stay green |
| `README.md`, `index.html`, `code/**`, `docs/*.png|jpg` outside `docs/3d` | cloud session | edit freely |
| `LICENSE*`, the credit name | Ao's decision | keep as they are, flag questions |
| `images/`, `labels/*` except texture_ita.csv, `prompts/` | frozen data | never edit |

Work on your own branch, not `main`. Before editing, merge the latest local-session branch (named in the brief
you were given, for example `3d-ladder`). Commit as `Ao Zuo <316852364+0vvI@users.noreply.github.com>`.

## Hard rules

- Don't make the repository public, enable GitHub Pages, or change visibility, collaborators or settings. Those
  steps are Ao's.
- Don't choose or change the data licence or the credit name. If the text needs a decision, write the question
  in your summary.
- The 3D images may not be described as anything they aren't:
  - Since 13 Nov 2026 every render uses its identity's fitted FLAME head shape (`fit/results/fits_final.npz`,
    shape only, neutral expression; local tools `pigment_skin/tools/fitgeo/`). It is a landmark fit to one
    photograph, not a scan or a likeness, and the back of the head and the neck come from the model, not the photo.
  - The skin texture is made from the identity's text description by recolouring a template texture. It is not
    made from the portrait.
  - Textures are made at 8192 px; the renders use 4096 px material maps. Say "4K material maps", not "8K".
  - Tone-ladder images: each skin on its own fitted head shape, with the same eyes, lashes, brow shape and
    lighting. Since 13 Nov every ladder skin uses its own colour, subdermal colour, scattering weight, specular,
    roughness, micro height and micro normal (all 4K); the scattering radius and micro-detail weight are one map
    for every identity; the mesh relief, eyes, lashes and ear relief are ID012's; the ear's colour tile is matched
    to each skin. The exposure is locked to the hero's for every skin, so brightness differences are the skin's.
  - There is no separate displacement map: relief at the scale of the render mesh (median edge 0.54 mm; up to
    0.35 mm deep) is baked into the mesh; finer relief is the micro height and normal maps, read as bump.
  - Texture tone and portrait tone are different measurements: unlit albedo against a studio-lit JPEG. Compare
    their orderings, not their levels (the darkest textures are far darker than their portraits).
- The pipeline code, `.blend` files and texture maps are not in this repo and will not be. Don't promise them.
- The head-shape fit (`fit/`) gives the renders their head shapes (since 13 Nov 2026). Never call a fitted head
  accurate, a reconstruction or a likeness: there is no ground-truth 3D shape, and the fit is scored against a
  detector's landmarks.
- FLAME model files and the MediaPipe model are never committed (`fit/assets/` is ignored). The landmark
  mapping, the offsets and the fitted coefficients are derived data and are committed.
- Every number about the fit comes from `python fit/report.py`, which reads `fit/results/summary.json` and
  `solver_agreement.json`. Paste from its output; don't retype.
- Every number in the README must come from a file in this repo. Recompute it, don't copy it.

## Facts you can rely on

- Median cheek ITA per prompted tone, computed from `labels/measurements_v1.csv` (10 values per tone, stored to
  one decimal, so a median can land exactly on a half): 65.50, 55.35, 53.25, 51.50, 52.00, 51.40, 44.40, 36.55,
  25.40, 4.95. Quote them to two decimals (or state the rounding rule), so any reader reproduces them from the
  CSV. The 4 Nov table (53.3, 36.6, 5.0) is the same data rounded half-up; it was not wrong.
- No prompt names a tone or a number. Each Monk tone is written as a phrase (`skin_words` in
  `labels/identities_v0.csv`, e.g. tone 3 "light skin with warm undertone", tone 6 "medium-tan skin"). Say "the
  phrases for tones 2–6 barely change the measured ITA", not "naming a middle tone".
- YuNet (`code/face_detection_yunet_2023mar.onnx`) is MIT licensed. Any comment that says Apache-2.0 is wrong.
- FLAME 2023 Open is CC BY 4.0. The 3D renders use its head shape, so the README must credit it:
  "FLAME 2023 Open head model (Li et al., Max Planck Institute for Intelligent Systems), CC BY 4.0".
- 31 of the 100 identities have a skin texture that passed every automatic gate. They cover prompted tones 1–9
  (none at 10). `labels/texture_ita.csv` gives each one's cheek ITA measured on the texture: the 8192 px texture
  reduced to 2048, two 0.07 W square patches centred at UV (0.325, 0.54) and (0.675, 0.54) (top-down), median
  CIELAB, ITA averaged over the two. On these 31, Spearman between prompted tone and texture ITA is −0.96; between
  prompted tone and portrait ITA it is −0.83 (re-derived independently on 5 Nov; recompute before quoting).
- `docs/3d/renders.csv` lists what is in `docs/3d/tone_ladder.jpg`, column by column. Its `file` column names the
  local render, which is not in this repo.
- `docs/3d/ID068_*` (prompted tone 7) and `docs/3d/ID089_*` (tone 9) are full heads: the whole pipeline run on
  that identity's texture, so its identity-dependent skin maps are its own: colour, subdermal colour, scattering
  weight, specular, micro height and normal, roughness, and the ear tiles, plus its own eyes, lashes and mesh
  relief. The scattering-radius and micro-detail-weight maps are generic, the same on every head. Each sits on its
  own fitted head shape. Shared with the hero: the eye and brow method (the brow shape is a stock mask, not the
  identity's), the lights, and the hero's exposure for each shot. They have no turntable. Their cheek ITA is not in any repo file, so don't quote one.
- `docs/3d/texture_vs_portrait_ita.png` plots `labels/texture_ita.csv` (texture and portrait ITA against
  prompted tone, the 31 textures).
- `docs/3d/render_vs_texture_ita.png` plots `docs/3d/render_vs_texture_ita.csv`: the cheek ITA of the front render
  (H01) of each of the seven ladder skins, against its texture's and its portrait's (render patches placed by
  YuNet, as in `code/measure.py`). Since 13 Nov these are the fitted-head renders. The renders keep the textures'
  order exactly (Spearman 1.0 over 7), but they read lighter than the texture: by +6.7° to +12.2° for the four
  lightest skins (ID004 +6.7, ID056 +9.9, ID012 +11.6, ID074 +12.2) and by more for the three darkest (ID068 +15.1,
  ID077 +16.0, ID089 +29.8). Say that. Don't call the renders tone-accurate. The head shape alone moves a render's
  ITA: ID012 with the same maps read 53.3° on the old stock head and 56.9° on its fitted head. Against the portraits the order differs (Spearman 0.89, the same as texture against
  portrait): ID068 and ID077 render darker than ID074 although their portraits are lighter, because a render's
  tone comes from its texture.
- The gallery link (GitHub Pages) works only after Ao makes the repo public and turns Pages on.
