# Dataset Card

## Dataset summary

KeyVision-QA does not distribute a real industrial dataset in v0.1.0. It provides a deterministic
synthetic generator solely for integration testing, visualization, CI, and documentation examples.

## Synthetic example data

- **Source:** generated locally by `keyvision.data.synthetic`
- **License:** generator code is MIT; generated geometric examples may be used under the same terms
- **Content:** stylized keyboard grids with one injected defect per image
- **Classes:** missing keycap, misaligned keycap, print defect, stain, scratch, foreign object
- **Disclosure:** every image contains a visible `SYNTHETIC EXAMPLE - NOT PRODUCTION DATA` banner
- **Prohibited interpretation:** it does not approximate factory variation or validate accuracy

## Intended uses

- Validate file and label contracts
- Exercise training, inference, evaluation, ONNX, and UI paths
- Demonstrate reproducible splitting and error handling

## Out-of-scope uses

- Quoting industrial model quality
- Training a deployable inspection model
- Comparing architectures for production selection
- Representing a real keyboard brand, factory, customer, or workforce

## Real-data requirements

A future dataset should document capture hardware, optical geometry, line speed, defect creation and
review protocol, annotation agreement, product/lot grouping, licensing, retention, access controls,
and train/validation/test leakage checks. Split by product lot or capture session where possible,
not by nearly adjacent frames.

## Known biases and risks

The generator has limited textures, perfectly controlled geometry, simple defect rendering, and no
real sensor noise. A model can overfit generator artifacts or the watermark. Those characteristics
make synthetic metrics unfit for deployment decisions.

## Public-data investigation

On 2026-07-30, the following candidates were reviewed with task and license boundaries:

- [Roboflow Keyboard Defect Detection v7](https://universe.roboflow.com/rsi-fcy8m/keyboard-defect-detection/dataset/7)
  lists 1,110 images and detection annotations, but its public version page does not state a
  redistributable dataset license.
- [Kaggle Keyboards Detection Dataset](https://www.kaggle.com/datasets/lorencjan/keyboards-detection-dataset)
  is keyboard-oriented rather than a verified defect taxonomy, and the accessible public page did
  not expose a license suitable for this release.
- [Keyboard Detection v2](https://universe.roboflow.com/keyboard-detection-v2/keyboard-detection-v2)
  declares CC BY 4.0 and contains keyboard/key-layout labels. It is a possible layout-localization
  or representation-pretraining source, but it is not a keyboard-defect benchmark and cannot
  support defect-accuracy claims.
- [Visual Anomaly (VisA)](https://registry.opendata.aws/visa/)
  is an AWS Open Data industrial anomaly dataset under CC BY 4.0 with 10,821 images and pixel-level
  annotations. The official archive is about 1.8 GB. It is a legitimate real-world anomaly
  benchmark, but it is not keyboard-specific and must be reported as a separate transfer test.
- [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
  contains more than 5,000 industrial anomaly images with pixel masks under CC BY-NC-SA 4.0. Its
  non-commercial and share-alike terms require separate handling, so no copy is included here.


No images, annotations, weights, or metrics from these candidates are copied into this repository.
