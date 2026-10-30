from __future__ import annotations

import cv2
import numpy as np

REASONS = {
    "correctly_worn": "Mask appears to cover the nose and mouth region.",
    "nose_exposed": "Nose region appears uncovered by the mask.",
    "mouth_exposed": "Mouth region appears uncovered by the mask.",
    "below_chin": "Mask appears to be positioned below the chin.",
    "no_mask": "No face mask is detected on the lower face.",
}


def make_explanation(class_name: str, landmarks: dict, quality: dict, uncertainty: float) -> str:
    reason = REASONS.get(class_name, "Predicted class from the mask-position classifier.")
    if not landmarks:
        reason += " Facial landmarks were unavailable, so the result should be reviewed."
    if quality.get("occlusion_risk", 0.0) >= 0.6:
        reason += " Possible occlusion or difficult face geometry was detected."
    if uncertainty >= 0.43:
        reason += " Model uncertainty is high; collect more/clearer frames."
    return reason


def draw_landmarks(face_crop: np.ndarray, points: dict[str, tuple[int, int]]) -> np.ndarray:
    output = face_crop.copy()
    for name, (x, y) in points.items():
        cv2.circle(output, (x, y), 3, (0, 255, 255), -1)
        if name in {"nose_tip", "upper_lip", "lower_lip"}:
            cv2.putText(output, name.replace("_", " "), (x + 4, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 255, 255), 1, cv2.LINE_AA)
    return output


def gradcam_overlay(model, transform, device: str, face_bgr: np.ndarray, class_index: int | None = None) -> np.ndarray:
    """Create a Grad-CAM heatmap from the last MobileNetV3 feature block."""
    import torch
    from PIL import Image

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    activations, gradients = {}, {}
    target_layer = model.features[-1]

    def capture_forward(_module, _inputs, output):
        activations["value"] = output

    def capture_backward(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(capture_forward)
    backward_handle = target_layer.register_full_backward_hook(capture_backward)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        if class_index is None:
            class_index = int(logits.argmax(1).item())
        logits[0, class_index].backward()
        feature_maps = activations["value"].detach()[0]
        grads = gradients["value"].detach()[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=0)).cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()

    cam = cv2.resize(cam, (face_bgr.shape[1], face_bgr.shape[0]))
    cam -= cam.min()
    cam /= max(float(cam.max()), 1e-8)
    heat = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(face_bgr, 0.58, heat, 0.42, 0)
