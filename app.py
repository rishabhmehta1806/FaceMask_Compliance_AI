from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from maskguard.config import load_config
from maskguard.pipeline import MaskCompliancePipeline
from maskguard.explainability import gradcam_overlay

st.set_page_config(page_title="MaskGuard AI", page_icon="😷", layout="wide")
st.title("MaskGuard AI — Face Mask Compliance")
st.caption("YOLO face detection • MediaPipe landmarks • mask-position classifier • quality • uncertainty • temporal logic")
st.info("For a real-time webcam or CCTV stream, run `python run_camera.py --source 0` in a terminal. This page analyses uploaded images.")

@st.cache_resource
def get_pipeline():
    return MaskCompliancePipeline(load_config())

uploaded = st.file_uploader("Upload a frame or image", type=["jpg", "jpeg", "png", "webp"])
show_gradcam = st.checkbox("Show Grad-CAM explanation heatmap", value=False)
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    try:
        pipeline = get_pipeline()
        pipeline.tracker.tracks.clear()  # still images must not share temporal history
        annotated, results = pipeline.process_frame(frame)
        left, right = st.columns([1.25, 1])
        with left:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated result", use_container_width=True)
        with right:
            if not results:
                st.warning("No face detected in this image.")
            for result in results:
                st.subheader(f"Face track {result['track_id']}")
                st.metric("Mask classification", result["decision"])
                st.metric("Compliance score", f"{result['compliance_score']:.1f}/100")
                st.metric("Uncertainty", f"{result['mean_uncertainty']:.2f}")
                st.write("**Explanation:**", result["reason"])
                st.write("**Image quality:**", {k: round(v, 3) if isinstance(v, float) else v for k, v in result["quality"].items()})
                st.caption("Temporal persistence is evaluated only in the live webcam/video runner, not from a single uploaded still image.")
                if show_gradcam:
                    heatmap = gradcam_overlay(pipeline.classifier.model, pipeline.classifier.transform,
                                              pipeline.classifier.device, result["crop"],
                                              result["predicted_index"])
                    st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="Grad-CAM: regions influencing the predicted class", use_container_width=True)
                st.write("**Class probabilities:**")
                probs = result["probabilities_temporal"]
                st.bar_chart({name: float(probs[i]) for i, name in enumerate(pipeline.classifier.class_names)})
    except Exception as exc:
        st.error(str(exc))
        st.markdown("Complete the setup and training steps in `README.md` before running the app.")
