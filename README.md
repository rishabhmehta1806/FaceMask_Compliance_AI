# MaskGuard AI — Explainable, Uncertainty-Aware Face Mask Compliance

**Suggested research title:** *An Explainable, Uncertainty-Aware and Temporally Consistent Framework for Real-Time Face Mask Compliance Detection Under Challenging Visual Conditions*

A complete Python project scaffold for real-time face-mask position monitoring. It combines:

1. **YOLO face detection** — detects face bounding boxes in each camera frame.
2. **MediaPipe FaceMesh** — localizes nose, upper/lower lip, mouth corners and chin landmarks.
3. **Five-class mask-position classification** — correctly worn, nose exposed, mouth exposed, below chin, and no mask.
4. **Image quality assessment** — blur, illumination, face size and landmark-availability/occlusion proxy.
5. **Temporal verification** — tracks faces across frames, smooths probabilities and requires a persistent violation before alerting.
6. **Uncertainty estimation** — normalized entropy, confidence and quality determine whether the system should return a decision or ask for a clearer frame.
7. **Explainability** — human-readable reason plus optional Grad-CAM heatmap.
8. **Compliance scoring** — combines correct-wear probability, temporal correct ratio, image quality and confidence.
9. **Alert logging** — a violation that remains confident and improper for five continuous seconds is logged in `logs/alerts.csv`.

> **Important model-status note:** This ZIP contains the complete source code and reproducible training pipeline. It does **not** contain a fabricated or randomly initialized “trained” mask classifier. A real five-class checkpoint is created only after training on labelled images. The YOLO face detector is a separate pretrained checkpoint downloaded by the setup script. I could not download the large training dataset or run the five-class training job in this environment, so there are no measured accuracy claims included.

---

## 1. Project structure

```text
FaceMask_Compliance_AI/
├── app.py                         # Streamlit image demo + probability chart + Grad-CAM
├── run_camera.py                  # Webcam / video / CCTV stream inference
├── train.py                       # MobileNetV3 transfer-learning training
├── evaluate.py                    # External labelled-test evaluation
├── config.yaml                    # thresholds, paths and class configuration
├── requirements.txt
├── scripts/
│   ├── download_face_model.py     # downloads YOLOv8n-face weights
│   ├── import_maskedfacenet.py    # imports compatible MaskedFace-Net subsets
│   ├── prepare_dataset.py         # source-group-aware train/val/test split
│   └── label_images.py            # keyboard-based manual image labeller
├── src/maskguard/
│   ├── detector.py                # YOLO + MediaPipe detector fallback
│   ├── landmarks.py               # MediaPipe FaceMesh landmarks
│   ├── classifier.py              # trained MobileNetV3 classifier loader
│   ├── quality.py                 # blur / illumination / face-size / landmark metrics
│   ├── uncertainty.py             # entropy- and confidence-based uncertainty
│   ├── temporal.py                # IoU tracking, smoothing and 5-second alert logic
│   ├── explainability.py          # explanations and Grad-CAM
│   └── pipeline.py                # end-to-end per-frame orchestration
├── docs/
│   ├── PROJECT_REPORT.md          # report-ready project documentation
│   └── ARCHITECTURE.md             # architecture and formulas
├── data/input/                    # labelled class folders go here
├── data/processed/                # generated train/val/test folders
├── models/                        # detector + trained classifier checkpoints
├── reports/                       # evaluation.json and confusion matrix
└── logs/                           # runtime alert CSV
```

## 2. Environment setup

Use Python **3.10–3.12**. A GPU is helpful but not mandatory. The code automatically selects CUDA, Apple Silicon MPS, or CPU for PyTorch.

### macOS / Linux

```bash
cd FaceMask_Compliance_AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd FaceMask_Compliance_AI
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the pretrained YOLO face detector:

```bash
python scripts/download_face_model.py
```

This saves `models/yolov8n-face.pt`. The download is from the YOLO Face project by akanametov. If this checkpoint cannot be loaded on your installed Ultralytics version, use the included MediaPipe fallback temporarily or use a compatible checkpoint from the source project. For the architecture to use YOLO as specified, confirm the startup message says `Face detector backend: YOLO`.

## 3. Prepare the five-class dataset

The required input folder structure is:

```text
data/input/
├── correctly_worn/
├── nose_exposed/
├── mouth_exposed/
├── below_chin/
└── no_mask/
```

Place face crops in the matching folders. You can use the bundled keyboard labeller for a folder of unlabelled face crops:

```bash
python scripts/label_images.py --source path/to/unlabelled_face_crops --destination data/input
```

Keys: `1` correctly worn, `2` nose exposed, `3` mouth exposed, `4` below chin, `5` no mask, `s` skip, `q` quit. Label only images you have permission to use. For consistent labels, define the classes before annotation and have ambiguous samples reviewed by a second annotator.

### Importing MaskedFace-Net

MaskedFace-Net is a useful starting point because it contains correctly masked and incorrectly masked face images. Its original release contains two large parts: CMFD (correctly masked) and IMFD (incorrectly masked). The project importer maps known subsets as follows:

| MaskedFace-Net subset | Project class |
|---|---|
| CMFD | `correctly_worn` |
| IMFD2 (nose uncovered) | `nose_exposed` |
| IMFD3 (mouth uncovered) | `mouth_exposed` |
| IMFD1 (chin uncovered) | **Skipped** — not the same as a mask positioned below the chin |

After extracting the dataset, run:

```bash
python scripts/import_maskedfacenet.py --source path/to/MaskedFace-Net --destination data/input
```

**Do not assume the importer can infer a subtype from an unsplit `IMFD` folder.** If the archive does not retain the IMFD1/2/3 folder labels, sort those examples using the original dataset metadata/documentation or label them manually. The importer intentionally does not guess.

MaskedFace-Net does not provide the full five classes required here: you must add real examples for `below_chin` and `no_mask`. Do not relabel IMFD1 as `below_chin`; it represents an uncovered-chin configuration, which is a different geometry. For a stronger research result, supplement synthetic images with diverse real photographs/video frames of different mask colours, materials, camera angles, distances and illumination. Keep the dataset subject-disjoint where possible.

The original repository states the dataset comprises 133,783 images and is distributed under **CC BY-NC-SA 4.0**; it is restricted to non-commercial use and requires attribution and share-alike terms for derivatives. Read the exact source licence and image-level FFHQ terms before redistribution or any commercial use.

## 4. Create train / validation / test splits

```bash
python scripts/prepare_dataset.py --input data/input --output data/processed --force
```

The splitter keeps files with the same source filename stem together across splits when possible. This helps reduce leakage for paired mask-position variants created from the same source face. If you are working with a dataset that changes the filenames for each version of the same person, perform a subject-ID-based split before training instead.

For a credible academic experiment, aim for **hundreds to thousands of diverse, correctly labelled images per class**, not just the three-image minimum that the split utility permits for a smoke test. Record class counts and the source of every class in your report.

## 5. Train the mask-position classifier

```bash
python train.py --epochs 40 --batch-size 32 --workers 2
```

For a low-memory laptop:

```bash
python train.py --epochs 40 --batch-size 8 --workers 0
```

Training details:

- MobileNetV3-Large transfer learning from ImageNet weights.
- Resize/crop, horizontal flip, moderate brightness/contrast/colour jitter and rotation augmentation.
- Weighted sampling to reduce class imbalance.
- AdamW optimiser, cosine learning-rate schedule, label smoothing and early stopping using validation macro-F1.
- Temperature scaling fitted on validation logits.
- Held-out test accuracy, macro-F1, class-wise precision/recall/F1 and confusion matrix saved after training.

On the first run, torchvision may download ImageNet backbone weights. If you intentionally want to train from random initialization, add `--scratch` (not recommended for a small student dataset).

Successful training creates:

- `models/best_mask_classifier.pth`
- `reports/evaluation.json`
- `reports/confusion_matrix.png`

The values in `evaluation.json` are generated by your actual run. Do not place estimated accuracy in the final report as if it were measured.

## 6. Run the project

### Webcam

```bash
python run_camera.py --source 0
```

### Video file

```bash
python run_camera.py --source path/to/test_video.mp4 --save runs/annotated_output.mp4
```

### IP camera / CCTV stream

Pass the RTSP or HTTP stream address as `--source` (quote it in the shell). Make sure you have permission to access and process that camera feed.

Press `q` in the OpenCV window to exit. Alerts are appended to `logs/alerts.csv` after a confident improper-wearing decision persists for the configured five seconds.

### Web interface

```bash
streamlit run app.py
```

Upload a still image to view annotated detections, quality metrics, uncertainty, class probabilities and optional Grad-CAM. Use `run_camera.py` for true temporal verification across live frames; an uploaded still image cannot demonstrate a five-second persistence check.

## 7. Main outputs and score interpretation

Per detected face, the application reports:

- predicted mask-position class;
- calibrated class probabilities and maximum confidence;
- uncertainty from normalized entropy, low confidence and quality penalty;
- blur, illumination, face-size and landmark availability metrics;
- temporal consistency;
- compliance score from 0 to 100;
- textual reason and optional Grad-CAM visual evidence.

The current score is a **research score**, not a legally certified compliance measure:

```text
Compliance = 100 × (
    0.50 × mean P(correctly_worn)
  + 0.20 × recent correct-prediction ratio
  + 0.20 × mean image-quality score
  + 0.10 × mean model confidence
)
```

A stable but incorrect prediction therefore does not become a high compliance score. Uncertain or low-quality frames are shown as `REVIEW` and do not produce a new violation alert.

## 8. Important limitations and responsible use

- The five-class classifier must be trained on your actual labelled dataset. It is not bundled with this ZIP.
- MediaPipe landmarks localize reference areas; they do not, on their own, prove that nose or mouth skin is exposed.
- The current occlusion value is a **landmark-availability proxy**, not a clinically validated occlusion detector.
- YOLO face boxes are associated across frames with a lightweight IoU tracker. Crowded scenes, rapid movement and overlapping faces can cause track switches; ByteTrack/DeepSORT can be explored as future work.
- MaskedFace-Net is synthetic/edited and mostly frontal; results on real CCTV must be evaluated separately. Report domain shift and limitations.
- Do not use the tool for identity recognition, punitive decisions, or solely automated enforcement. Provide human review and comply with local privacy, CCTV and data-protection requirements.

## 9. Tests

```bash
pip install pytest
pytest -q
```

## 10. Research references

1. A. Cabani, K. Hammoudi, H. Benhabiles, and M. Melkemi, “MaskedFace-Net — A Dataset of Correctly/Incorrectly Masked Face Images in the Context of COVID-19,” *Smart Health*, 2020. DOI: `10.1016/j.smhl.2020.100144`. Dataset: <https://github.com/cabani/MaskedFace-Net>.
2. YOLO Face pretrained face-detector repository: <https://github.com/akanametov/yolo-face>.
3. MediaPipe Face Landmarker / Face Mesh: <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>.
4. PyTorch / torchvision model documentation: <https://pytorch.org/vision/stable/models.html>.

See `docs/PROJECT_REPORT.md` for report-ready methodology and `docs/ARCHITECTURE.md` for the system diagram and equations.
