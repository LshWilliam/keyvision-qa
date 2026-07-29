# Failure-case analysis

Runtime error mining writes false positives and false negatives separately. False positives are
ranked by confidence so the most harmful model beliefs are reviewed first.

| Failure mode | Expected symptom | Diagnostic slice | Candidate mitigation |
| --- | --- | --- | --- |
| Specular reflection | bright regions resemble scratches or missing legends | light angle and key material | polarizer, diffuse lighting, reflection augmentation |
| Low contrast | stain or print damage disappears into keycap | contrast and color distance | controlled exposure, color normalization, hard-negative mining |
| Small target | short scratch or partial legend is missed | box area and feature level | higher resolution, tiled inference, anchor/feature tuning |
| Viewpoint change | boxes drift or anomaly map lights edges | camera and pose metadata | rigid fixture, registration, pose augmentation |
| Illumination change | global anomaly score rises | exposure, shift, time of day | reference refresh, photometric normalization, drift alarms |
| Occlusion | fingers, tools, or packaging trigger alerts | occlusion tag | line interlock, occlusion class, review policy |
| Domain shift | new layout or material causes widespread errors | SKU, lot, camera | grouped evaluation, adaptation, per-domain thresholds |

## Review procedure

1. Freeze the model, thresholds, and test split before review.
2. Separate taxonomy errors from localization errors.
3. Inspect duplicate frames and lot/session leakage.
4. Plot error rates by SKU, lot, camera, lighting recipe, defect size, and operator-independent
   metadata where collection is authorized.
5. Change one factor at a time and rerun the frozen evaluation.
6. Record whether the change trades false accepts against false rejects.

No real failure-case images are included because no authorized dataset is available.

