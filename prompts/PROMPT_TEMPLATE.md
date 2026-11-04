# Portrait prompt (v0, 18 Sep 2026)

One portrait per row of `labels/identities_v0.csv`. The constraints ask for a front-facing, evenly lit,
unretouched face with the forehead and ears visible, so that skin can be measured and compared across the set.
`code/build_prompts.py` fills the template and puts everything on one line, tagged `[HDS-IDxxx]`.

## Base portrait

```
Photorealistic studio portrait photograph of a {age_intended}-year-old {gender_presentation} of
{region_descriptor} heritage with {skin_words}. {pigment_feature_sentence}
{hair}, {facial_hair_sentence} forehead and both ears visible, no glasses, no jewellery, no makeup,
neutral relaxed expression, mouth closed, eyes open looking at the camera.
Head and shoulders, face centred and front-facing, camera at eye level, 85 mm lens.
Soft even diffuse lighting from the front, no strong shadows, no specular hot spots, plain mid-grey background.
Unretouched natural skin: pores, fine lines and natural colour variation visible. No skin smoothing.
```

Fill rules:
- `pigment_feature_sentence`: empty when `pigment_feature` is `none`; otherwise `Natural skin with {pigment_feature}.`
- `facial_hair_sentence`: empty when `facial_hair` is `none`; otherwise `{facial_hair},`.
- Keep `seed` with the row if the generator accepts one; otherwise record the generator's own id.
