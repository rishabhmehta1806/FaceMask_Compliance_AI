# Project viva and demo guide

## 60-second project explanation

“Our project is a real-time face-mask compliance system. First, a YOLO face detector finds face regions in the camera frame. MediaPipe FaceMesh localizes important points around the nose, mouth and chin. We crop each face and pass it to a MobileNetV3 classifier trained for five mask-wearing states: correctly worn, nose exposed, mouth exposed, below chin and no mask. Along with the prediction, we evaluate image quality and model uncertainty. If the frame is blurry, dark, too small or the model is unsure, we show REVIEW instead of making a strong claim. We then associate faces across frames using IoU and smooth the predictions over time. An alert is logged only when improper wearing remains confident for five seconds. For explainability, we show the predicted reason, facial reference landmarks and an optional Grad-CAM heatmap.”

## Common viva questions

**1. Why use YOLO?**  
YOLO performs object detection in a single forward pass and is suitable for locating face regions in video frames. We use a face-specific checkpoint rather than a generic COCO person detector.

**2. Why MediaPipe FaceMesh?**  
It provides facial reference points such as the nose, lips and chin. These help locate facial regions and display landmark evidence. It is not treated as the sole mask classifier.

**3. Why MobileNetV3?**  
It is a compact convolutional architecture suitable for transfer learning and real-time inference on laptops and edge devices.

**4. Why five classes instead of two?**  
A binary mask/no-mask decision cannot distinguish a mask worn under the nose or below the chin. Fine-grained classes give more actionable feedback.

**5. Why assess image quality?**  
Blur, poor illumination and very small faces can make predictions unreliable. Quality gating helps the system abstain rather than issue an overconfident alert.

**6. What is uncertainty estimation?**  
We combine normalized predictive entropy, the maximum class probability and image quality. Higher uncertainty means the system should request clearer frames or human review.

**7. Why temporal consistency?**  
A single frame can be misclassified due to blur, motion or occlusion. Short-term probability smoothing reduces one-frame fluctuations, while the alert timer requires a persistent state.

**8. Why five seconds?**  
It is a configurable prototype threshold that prevents immediate alerts from transient errors. It should be tuned using labelled video sequences and the desired response time.

**9. What is Grad-CAM?**  
Grad-CAM uses gradients flowing into a convolutional feature layer to highlight image regions that influenced the selected class score. It is a qualitative explanation aid, not proof of causation.

**10. How is compliance score calculated?**  
It is a weighted 0–100 prototype score using the temporal mean probability of correct wearing, the recent ratio of correct predictions, image quality and classifier confidence. It is not a regulatory standard.

**11. What are the main limitations?**  
The public MaskedFace-Net data has synthetic mask overlays and does not provide the exact five target classes. Real examples are needed for below-chin and no-mask classes, and real CCTV performance must be evaluated separately.

**12. Is the five-class model already trained?**  
The source includes the training pipeline, but the final five-class checkpoint is generated only after the dataset is prepared and `train.py` finishes. We report only the measured test metrics in `reports/evaluation.json`.

## Demo sequence

1. Run `python run_camera.py --source 0` and show YOLO face boxes.
2. Show facial landmark dots around the nose/mouth/chin.
3. Demonstrate the five mask states with labelled test examples.
4. Move to a low-light or blurred example and show `REVIEW`.
5. Keep a mask improperly worn for five seconds to demonstrate alert creation in `logs/alerts.csv`.
6. Open the Streamlit app and enable the Grad-CAM checkbox for one uploaded test image.
7. Finish by opening `reports/evaluation.json` and the confusion matrix; discuss class-wise errors rather than quoting only accuracy.
