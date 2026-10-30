# An Explainable, Uncertainty-Aware and Temporally Consistent Framework for Real-Time Face Mask Compliance Detection Under Challenging Visual Conditions

## Abstract

This project proposes a modular computer-vision framework for monitoring face-mask wearing position in camera frames. Unlike binary mask/no-mask systems, the proposed classifier distinguishes five states: correctly worn, nose exposed, mouth exposed, below chin and no mask. A YOLO-based face detector extracts face regions, while MediaPipe FaceMesh localizes facial reference points around the nose, mouth and chin. A MobileNetV3-Large classifier predicts mask position from each face crop. An image-quality module estimates blur, illumination, face size and landmark availability. The framework combines calibrated class probabilities with normalized predictive entropy to estimate uncertainty and abstains from a categorical decision when evidence is weak. A lightweight IoU tracker maintains short-term face tracks and smooths predictions over consecutive frames. A violation alert is generated only when a confident improper-wearing state persists for a configurable duration (five seconds by default). Explanations include a textual reason, facial reference landmarks and an optional Grad-CAM heatmap. A weighted compliance score summarizes position probability, temporal correctness, image quality and model confidence. The implementation includes data preparation, training, evaluation, webcam/video inference and alert logging. The framework is intended as a research prototype; performance must be established on a representative, held-out real-world dataset before deployment.

**Keywords:** face-mask compliance, YOLO, MediaPipe FaceMesh, MobileNetV3, temporal consistency, uncertainty estimation, Grad-CAM, image quality assessment, explainable AI.

## 1. Introduction

Face-mask monitoring is a fine-grained visual classification problem. A binary system may recognize that a mask-like object is present while missing whether it covers the nose and mouth. In practice, a useful monitoring system should distinguish multiple wearing positions, handle difficult frames, and avoid triggering alerts from a single unreliable image.

The proposed system separates the task into face detection, facial reference localization, mask-position classification, image-quality assessment, uncertainty estimation, temporal verification and explanation. This modular approach makes individual components testable and allows the decision policy to abstain when the image is unsuitable.

## 2. Problem statement

Given a video stream \(V=\{I_1,I_2,\ldots,I_T\}\), detect each visible face, classify its mask-wearing state, estimate the reliability of the prediction, and issue an alert only when an improper state is sustained over time.

For each face crop, the classifier predicts one of five classes:

1. `correctly_worn` — mask covers the nose and mouth region;
2. `nose_exposed` — nose is uncovered;
3. `mouth_exposed` — mouth is uncovered;
4. `below_chin` — mask is positioned below the chin;
5. `no_mask` — no mask is worn.

The system must also produce a quality score \(Q\), uncertainty estimate \(U\), temporal consistency measure \(T_c\), compliance score \(S_c\), and an explanation for the predicted state.

## 3. Objectives

- Detect face regions in live camera frames using a YOLO face detector.
- Localize facial landmarks around the nose, mouth and chin using MediaPipe FaceMesh.
- Train a five-class mask-position classifier using transfer learning.
- Estimate blur, illumination, face-size suitability and landmark availability.
- Calibrate confidence and estimate uncertainty using predictive entropy.
- Track faces across frames and smooth short-term predictions.
- Generate an alert only after a confident improper state persists for five seconds.
- Explain decisions using a text reason and Grad-CAM visual evidence.
- Produce a 0–100 mask-compliance research score.
- Evaluate the model with accuracy, class-wise precision/recall/F1, macro-F1 and a confusion matrix.

## 4. Related work and dataset selection

Cabani et al. introduced MaskedFace-Net, a dataset of 133,783 images designed to represent correctly and incorrectly worn masks. It provides a useful starting point for mask-position recognition, with correctly masked and incorrectly masked subsets. The incorrectly masked cases include different mask-to-face placements, such as uncovered nose, uncovered mouth and uncovered chin configurations.

The dataset is derived from FFHQ face images with mask overlays. As a result, it is valuable for controlled experiments but does not fully represent real CCTV conditions, material variety, mask colours, perspective, motion blur or compression artefacts. The project therefore requires supplementary real-world examples for `below_chin` and `no_mask`, and strongly recommends adding real examples for every category.

**Label mapping caution:** the MaskedFace-Net IMFD1 case exposes the chin while the mask covers the nose and mouth. This is not equivalent to a mask positioned below the chin. The included importer maps only the clearly compatible CMFD, IMFD2 and IMFD3 subsets and skips IMFD1 rather than silently assigning the wrong label.

## 5. Proposed methodology

### 5.1 Face detection

A pretrained YOLO face detector processes the full frame and returns face bounding boxes:

\[
B_t=\{(x_1,y_1,x_2,y_2)_i\}_{i=1}^{N_t}
\]

A small margin is added around each box to preserve the lower-face region. The project includes a MediaPipe fallback for development when the YOLO checkpoint is unavailable; final YOLO-based experiments should verify that the runtime backend reports YOLO.

### 5.2 Facial landmark localization

MediaPipe FaceMesh estimates dense facial landmarks. The implementation extracts reference points for the nose tip, upper lip, lower lip, mouth corners and chin. These points provide region localization and support visual explanation. The landmarks are not treated as direct proof of mask coverage, because a landmark detector can fail or return a point even when the corresponding surface is partly occluded.

### 5.3 Mask-position classification

Each face crop is resized and normalized before being passed to a MobileNetV3-Large classifier. The final fully connected layer is replaced with a five-output classification head. ImageNet-pretrained weights are used for transfer learning unless the `--scratch` flag is supplied.

Training uses moderate geometric and colour augmentation, weighted sampling to reduce class imbalance, AdamW optimization, label smoothing, a cosine learning-rate schedule and early stopping based on validation macro-F1. The best checkpoint is selected using the validation set. The held-out test partition is used only after model selection.

### 5.4 Image-quality assessment

The quality module computes the following components:

- **Blur:** variance of the Laplacian, scaled to a bounded score.
- **Illumination:** a score based on the mean grayscale intensity relative to mid-range brightness.
- **Face size:** a bounded score based on the detected face crop dimensions.
- **Landmark availability:** a proxy indicating whether the key reference points were found.

The overall score is:

\[
Q=0.35B+0.25L+0.25S+0.15K
\]

where each component is normalized to \([0,1]\). The occlusion field is explicitly an approximate landmark-availability proxy; a dedicated occlusion segmentation model would be required for stronger occlusion claims.

### 5.5 Uncertainty estimation

Let \(p_i\) be the calibrated class probability and \(K\) the number of classes. The maximum class probability is confidence \(C\), while normalized entropy is:

\[
H(p)=-\frac{\sum_{i=1}^{K}p_i\log(p_i)}{\log K}
\]

The prototype uncertainty measure is:

\[
U=0.55H(p)+0.35(1-C)+0.10(1-Q)
\]

A prediction with low confidence, high entropy or poor quality is flagged as `REVIEW`. This is an abstention mechanism: it avoids treating every frame as sufficiently reliable for a compliance decision. Temperature scaling is fitted using validation logits before deployment.

### 5.6 Temporal consistency and alert policy

The tracker associates face boxes between frames using Intersection over Union (IoU). A short rolling history stores the probability vector from each frame. Temporal probabilities are averaged:

\[
\bar p_t=\frac{1}{N}\sum_{j=t-N+1}^{t}p_j
\]

The smoothed class is selected from \(\bar p_t\). A violation timer starts when the smoothed state is improper and the current frame is not uncertain and has sufficient image quality. The timer resets when the state becomes correct or evidence becomes unreliable. When the improper state persists for at least five seconds, the event is written to `logs/alerts.csv`.

### 5.7 Explainable AI

The system generates a concise class-based explanation (for example, “Nose region appears uncovered by the mask”), with an additional note when landmarks are unavailable or image uncertainty is high. Optional Grad-CAM highlights image regions that influence the MobileNetV3 prediction. Grad-CAM is a qualitative debugging aid, not a causal proof of the model's reasoning.

### 5.8 Compliance score

The compliance score is calculated as:

\[
S_c=100(0.50P_c+0.20R_c+0.20Q+0.10C)
\]

where:

- \(P_c\): temporally averaged probability of the `correctly_worn` class;
- \(R_c\): fraction of recent frames predicted as `correctly_worn`;
- \(Q\): mean image-quality score;
- \(C\): mean model confidence.

The coefficients are initial design weights. They should be tuned on a validation set, and the effect of each term should be reported through an ablation study. The score is a prototype indicator, not a regulatory certification.

## 6. System implementation

The project uses Python, PyTorch, torchvision, Ultralytics, OpenCV, MediaPipe, scikit-learn, Matplotlib and Streamlit. It includes:

- `train.py` — classifier training and test reporting;
- `evaluate.py` — evaluation on an additional labelled folder;
- `run_camera.py` — webcam, video and stream inference;
- `app.py` — image-upload demo with Grad-CAM option;
- `scripts/import_maskedfacenet.py` — supported dataset subset importer;
- `scripts/prepare_dataset.py` — group-aware train/validation/test split;
- `scripts/label_images.py` — manual class labelling utility.

The YOLO checkpoint is downloaded separately. The mask classifier checkpoint is generated after training and saved in `models/best_mask_classifier.pth`.

## 7. Experimental protocol

A final academic experiment should report:

1. Dataset source and licence for every class.
2. Number of images per class before and after splitting.
3. Image resolution, augmentation and preprocessing settings.
4. Training device, number of epochs, batch size, learning rate and early stopping epoch.
5. Test accuracy and macro-F1, plus precision, recall and F1 per class.
6. Confusion matrix and representative failure cases.
7. Quality-stratified performance (clear, blurred, dim, small-face and partial-occlusion frames).
8. Calibration metrics such as Expected Calibration Error (ECE) and reliability diagrams (to be added for a fuller study).
9. Temporal false-alert rate, detection delay and track-switch rate on labelled video sequences.
10. Ablation results: without landmarks, without quality gating, without uncertainty, without temporal smoothing and without Grad-CAM review.

The code writes actual held-out test results to `reports/evaluation.json` and `reports/confusion_matrix.png`. No accuracy value is prefilled in this report because no model was trained in the build environment.

## 8. Expected outcomes

The expected output is a real-time annotated stream showing face boxes, track identifiers, predicted mask state and compliance score. It should return `REVIEW` for ambiguous frames and log only persistent confident improper-wearing events. The training/evaluation tools support reproducible comparisons across model versions.

The word “expected” is intentional: the quality of the final result depends on the size, balance, annotation accuracy and real-world diversity of the training and test data. No numerical performance claim should be made until a model has been trained and evaluated.

## 9. Limitations

- The provided five-class model needs a user-supplied labelled dataset and is not pretrained in the archive.
- MaskedFace-Net mask overlays are synthetic and may cause domain shift in real camera feeds.
- The exact below-chin and no-mask categories require additional labelled data.
- MediaPipe landmarks can fail under occlusion, profile pose, blur or low resolution.
- The lightweight IoU tracker can swap IDs in dense crowds; a dedicated multi-object tracker may improve this.
- The uncertainty score is a practical heuristic; it should be validated and calibrated using held-out real-world data.
- The compliance score weights are design choices, not a published standard.
- Grad-CAM is qualitative and does not guarantee faithful causal explanations.

## 10. Future scope

- Add an occlusion segmentation branch for nose/mouth visibility.
- Use a temporal model (GRU, temporal convolution or video transformer) instead of simple rolling probability averaging.
- Replace greedy IoU matching with ByteTrack or DeepSORT.
- Add ECE, Brier score, reliability diagrams and selective-risk curves.
- Train and evaluate across multiple cameras, lighting conditions and geographic populations.
- Quantize the classifier and optimize the detector for edge hardware.
- Add a dashboard with aggregate counts while avoiding identity recognition and unnecessary storage of face images.

## References

1. A. Cabani, K. Hammoudi, H. Benhabiles, and M. Melkemi, “MaskedFace-Net — A Dataset of Correctly/Incorrectly Masked Face Images in the Context of COVID-19,” *Smart Health*, 2020. DOI: `10.1016/j.smhl.2020.100144`. <https://github.com/cabani/MaskedFace-Net>
2. YOLO Face project, pretrained face detector models and source. <https://github.com/akanametov/yolo-face>
3. Google MediaPipe Face Landmarker documentation. <https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker>
4. PyTorch torchvision model documentation. <https://pytorch.org/vision/stable/models.html>
