# 3D skin materials, batch 2: five more skins on the ladder

The first tone ladder (`docs/3d/`) has skins at prompted tones 1, 2, 6, 7, 8, 8 and 9, so it jumps from tone 2
to tone 6. Batch 2 renders five more gated skin textures the same way, adding tones 3, 4 and 5 plus a
second tone-1 and tone-7 skin. On 13 Nov 2026 every render in `docs/3d/`, these included, was redone on the
identity's fitted head shape.

![five more identities' skin, each on its own fitted head](tone_ladder.jpg)

*Five identities, each on its own fitted head shape, lightest to deepest texture cheek ITA: ID003, ID024, ID034,
ID050, ID065. Rows: front (H01), three-quarter (H03), cheek close-up (C01).*

## How they were made

These are ladder heads, like the seven in the first ladder. Each sits on its own head shape from the head-shape
fit (since 13 Nov 2026; shape only, neutral expression). The eyes, lashes, brow shape, mesh relief, lights and
per-shot exposure are ID012's (ID012e). The skin maps are the identity's own: colour, subdermal colour,
scattering weight, specular, roughness, micro height and micro normal, plus the ear's colour tile matched to each
skin (scattering radius and micro-detail weight are one map for every identity). Each head has the same seven
shots as ID012: four lighting setups (H01–H04) and three close-ups (C01 cheek, C02 eye, C03 lips). The
renders are Blender Cycles at 128 samples, 1200×1500 (heads) and 1024×1024 (close-ups); see `renders.csv`.

| Head | Prompted tone | Description (from `labels/identities_v0.csv`) | Portrait ITA | Texture ITA | Render ITA |
|:---:|:---:|:---|:---:|:---:|:---:|
| ID003 | 1 | woman, 38, East Asian, "very light, pale skin" | +70.7° | +70.2° | +75.0° |
| ID024 | 3 | man, 31, Southern European, "light skin with warm undertone" | +50.6° | +42.6° | +53.1° |
| ID034 | 4 | man, 34, mixed heritage, "light-medium skin" | +50.1° | +37.3° | +45.7° |
| ID050 | 5 | man, 84, Southeast Asian, "medium skin" | +42.8° | +27.6° | +38.1° |
| ID065 | 7 | woman, 57, South Asian, "tan-brown skin" | +47.0° | +26.0° | +36.4° |

Per head: [ID003](ID003_renders.jpg) · [close-ups](ID003_closeups.jpg) —
[ID024](ID024_renders.jpg) · [close-ups](ID024_closeups.jpg) —
[ID034](ID034_renders.jpg) · [close-ups](ID034_closeups.jpg) —
[ID050](ID050_renders.jpg) · [close-ups](ID050_closeups.jpg) —
[ID065](ID065_renders.jpg) · [close-ups](ID065_closeups.jpg)

## Tone

![render cheek ITA against texture cheek ITA](render_vs_texture_ita.png)

The front renders are measured like the portraits (YuNet at 640 px, two cheek patches;
`render_vs_texture_ita.csv`). They keep the textures' order and read +4.8° to +10.5° lighter than their textures:
about as much as the four lightest skins of the first ladder (+6.7° to +12.2°), and least for the lightest, ID003.

## Limits

- The head shape is a landmark fit to one photograph, not a 3D likeness of the portrait. The two women (ID003,
  ID065) still read masculine on their fitted heads: the brows, lids and stubble come from the shared template
  texture, and the eyes and lashes are ID012's.
- Texture and portrait ITA are different measurements (an unlit colour map against a studio-lit portrait), so compare
  orderings, not levels: over these five the order differs once (Spearman 0.9; ID065's portrait is lighter than
  ID050's, its texture darker), because the texture comes from the text description, not from the portrait.
- Some stubble comes with the shared template skin: it shows on ID024 and ID034 and faintly on ID065, whose
  descriptions have no facial hair. (ID050's moustache is in his description.)
