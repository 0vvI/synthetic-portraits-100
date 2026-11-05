# Notes for Claude sessions working on this repository

This is a portfolio project for Ao Zuo's CV. It should look good, and every number in it must be correct. Two
kinds of session work on it:

- **The local session** on Ao's Mac. It has the full-size portraits, the private texture and head pipeline,
  Blender, and the 31 skin textures. It produces everything under `docs/3d/`, plus `labels/texture_ita.csv`.
- **Cloud sessions** (this file is mostly for you). You have only this repository. You own the write-up and the
  site: `README.md`, `index.html`, `code/`, `LICENSE`, `LICENSE-DATA`.

## Who edits what

| Path | Owner | Cloud sessions |
|---|---|---|
| `docs/3d/**`, `labels/texture_ita.csv`, `CLAUDE.md` | local session | read, link, describe; never edit or regenerate |
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
  - The head shape is one generic FLAME 2023 Open head for every skin, not a fit to the portrait.
  - The skin texture is made from the identity's text description by recolouring a template texture. It is not
    made from the portrait.
  - Textures are made at 8192 px; the renders use 4096 px material maps. Say "4K material maps", not "8K".
  - Tone-ladder images: the same head, eyes, brows and lighting for every skin. Only the identity-dependent skin
    maps change (colour, subdermal colour, scattering weight, specular; the ear's colour tile is matched to each
    skin). The exposure is locked to the hero's for every skin, so brightness differences are the skin's.
  - Texture tone and portrait tone are different measurements: unlit albedo against a studio-lit JPEG. Compare
    their orderings, not their levels (the darkest textures are far darker than their portraits).
- The pipeline code, `.blend` files and texture maps are not in this repo and will not be. Don't promise them.
- Every number in the README must come from a file in this repo. Recompute it, don't copy it.

## Facts you can rely on

- Median cheek ITA per prompted tone (from `labels/measurements_v1.csv`): 65.5, 55.4, 53.2, 51.5, 52.0, 51.4,
  44.4, 36.5, 25.4, 4.9 (unrounded 4.945). The 4 Nov table had 53.3, 36.6 and 5.0; those three cells were wrong.
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
- The gallery link (GitHub Pages) works only after Ao makes the repo public and turns Pages on.
