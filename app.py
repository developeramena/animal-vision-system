from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CLASS_NAMES,
    DEFAULT_CLASS_NAMES_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_UNCERTAINTY_PATH,
)
from image_processing import encode_jpeg, quality_histogram, smart_process_image
from model_utils import (
    load_class_names,
    make_gradcam_heatmap,
    overlay_gradcam,
    predict_top_k,
)


st.set_page_config(
    page_title="Animal Vision System",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed",
)


MOBILE_CSS = """
<style>
:root {
    --forest: #075f43;
    --teal: #0a938f;
    --ink: #17211d;
    --muted: #5f6b66;
    --line: #dfe6e2;
    --success: #e8f7ed;
    --warning: #fff4d6;
}
.stApp { background: #ffffff; color: var(--ink); }
.block-container { max-width: 860px; padding-top: 4.4rem; padding-bottom: 5rem; }
h1, h2, h3 { color: var(--forest); letter-spacing: -0.02em; }
h1 { font-size: clamp(2rem, 7vw, 3rem) !important; margin-bottom: .25rem !important; }
h2 { font-size: clamp(1.45rem, 5vw, 2rem) !important; margin-top: 1.7rem !important; }
p, label, .stMarkdown { line-height: 1.55; }
div.stButton > button, div.stDownloadButton > button {
    min-height: 3rem;
    width: 100%;
    border-radius: .65rem;
    border: 1px solid var(--forest);
    font-weight: 700;
}
div.stButton > button[kind="primary"] {
    background: var(--forest);
    color: white;
}
[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #7da799;
    background: #f8fbf9;
    border-radius: .75rem;
    min-height: 8rem;
}
[data-testid="stMetric"] {
    border-top: 1px solid var(--line);
    padding-top: .6rem;
}
div[role="radiogroup"] { gap: .35rem; }
div[role="radiogroup"] label {
    border: 1px solid var(--line);
    border-radius: .6rem;
    padding: .55rem .75rem;
    min-height: 2.7rem;
    justify-content: center;
}
.quality-good, .quality-warning {
    padding: .9rem 1rem;
    border-radius: .65rem;
    margin: .5rem 0 1rem;
    font-weight: 650;
}
.quality-good { background: var(--success); color: #075f2c; }
.quality-warning { background: var(--warning); color: #744d00; }
.top-probability { margin: .3rem 0 1rem; }
.top-probability strong { display: block; margin-bottom: .2rem; }
.progress-track { height: .72rem; background: #e4e8e6; border-radius: 99px; overflow: hidden; }
.progress-fill { height: 100%; background: var(--forest); border-radius: 99px; }
.section-divider { border-top: 1px solid var(--line); margin: 1.4rem 0; }
.app-header {
    background: var(--forest);
    color: white;
    margin: -1rem 0 1rem;
    padding: 1rem 1.2rem;
    border-radius: .75rem;
    font-size: clamp(1.35rem, 5vw, 2rem);
    font-weight: 800;
    letter-spacing: -0.02em;
}
@media (max-width: 640px) {
    .block-container { padding: 4.1rem .9rem 4.5rem; }
    [data-testid="column"] { min-width: 100% !important; width: 100% !important; }
    div[role="radiogroup"] { display: grid !important; grid-template-columns: 1fr 1fr; }
    h1 { line-height: 1.08 !important; }
}
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    if not DEFAULT_MODEL_PATH.exists() or not DEFAULT_CLASS_NAMES_PATH.exists():
        return None, CLASS_NAMES, 0.60
    import tensorflow as tf

    model = tf.keras.models.load_model(DEFAULT_MODEL_PATH)
    class_names = load_class_names(DEFAULT_CLASS_NAMES_PATH)
    threshold = 0.60
    if DEFAULT_UNCERTAINTY_PATH.exists():
        threshold = float(
            json.loads(DEFAULT_UNCERTAINTY_PATH.read_text(encoding="utf-8")).get(
                "confidence_threshold", threshold
            )
        )
    return model, class_names, threshold


def show_metrics(result):
    before = result.original_metrics
    columns = st.columns(4)
    columns[0].metric("Brightness", f"{before.brightness:.1f}")
    columns[1].metric("Contrast", f"{before.contrast:.1f}")
    columns[2].metric("Blur score", f"{before.blur_score:.1f}")
    columns[3].metric("Resolution", f"{before.width}×{before.height}")
    if result.detected_problems:
        st.markdown(
            '<div class="quality-warning">Detected: '
            + ", ".join(problem.replace("_", " ") for problem in result.detected_problems)
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="quality-good">Image quality is suitable for prediction.</div>',
            unsafe_allow_html=True,
        )
    st.caption("Applied methods: " + (", ".join(result.applied_methods) or "No correction needed"))


def show_before_after(result):
    original_column, processed_column = st.columns(2)
    with original_column:
        st.subheader("Original")
        st.image(result.original_rgb, use_container_width=True)
    with processed_column:
        st.subheader("Processed")
        st.image(result.processed_rgb, use_container_width=True)


def show_top_predictions(top_predictions):
    st.subheader("Top 3 predictions")
    for item in top_predictions:
        probability_percent = item["probability"] * 100
        st.markdown(
            f"""
            <div class="top-probability">
                <strong>{item['class_name'].title()} — {probability_percent:.2f}%</strong>
                <div class="progress-track"><div class="progress-fill" style="width:{probability_percent:.2f}%"></div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def home_page():
    st.title("Project Overview")
    st.write(
        "This project first checks and processes animal images, then uses an "
        "EfficientNetB0 model to classify ten animal species."
    )
    st.subheader("Supported classes")
    st.write(" · ".join(class_name.title() for class_name in CLASS_NAMES))
    st.subheader("Complete workflow")
    st.markdown(
        "1. Merge two animal datasets  \n"
        "2. Detect quality problems and duplicates  \n"
        "3. Create the cleaned dataset  \n"
        "4. Train and evaluate EfficientNetB0  \n"
        "5. Upload or capture an image for prediction"
    )


def process_page():
    st.title("Image Processing")
    st.write("Upload one or more images to analyze and process them.")
    files = st.file_uploader(
        "Choose animal images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if not files:
        st.info("Upload images to start the quality analysis.")
        return

    report_rows = []
    for file in files:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader(file.name)
        try:
            result = smart_process_image(file.getvalue())
            show_metrics(result)
            show_before_after(result)
            st.pyplot(quality_histogram(result.original_rgb), use_container_width=True)
            report_rows.append({"filename": file.name, **result.report()})
            st.download_button(
                "Download processed image",
                data=encode_jpeg(result.processed_rgb),
                file_name=f"processed_{Path(file.name).stem}.jpg",
                mime="image/jpeg",
                key=f"download_{file.name}",
            )
        except Exception as error:
            st.error(f"Could not process {file.name}: {error}")
    if report_rows:
        report_frame = pd.DataFrame(report_rows)
        st.download_button(
            "Download quality report CSV",
            data=report_frame.to_csv(index=False).encode("utf-8"),
            file_name="image_quality_report.csv",
            mime="text/csv",
        )


def prediction_page():
    st.title("Animal Prediction")
    st.write("Upload or capture an animal image.")
    input_method = st.radio("Input method", ["Choose image", "Use camera"], horizontal=True)
    if input_method == "Choose image":
        image_file = st.file_uploader(
            "Choose image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="predict_upload"
        )
    else:
        image_file = st.camera_input("Use camera")
    if image_file is None:
        st.info("Choose an image or take a photo to continue.")
        return

    try:
        result = smart_process_image(image_file.getvalue())
    except Exception as error:
        st.error(f"The image could not be read: {error}")
        return

    st.subheader("Image quality")
    show_metrics(result)
    show_before_after(result)

    model, class_names, confidence_threshold = load_model_bundle()
    if model is None:
        st.warning(
            "Image processing is complete, but prediction is disabled because "
            "models/best_model.keras and models/class_names.json are not available."
        )
        return

    if result.final_status in {"Rejected", "Quarantined"}:
        st.error("This image is not suitable for model prediction.")
        return

    with st.spinner("Running EfficientNetB0 prediction..."):
        top_predictions = predict_top_k(model, result.processed_rgb, class_names, top_k=3)
    top_prediction = top_predictions[0]
    st.subheader("Prediction")
    st.markdown(f"## {top_prediction['class_name'].title()}")
    st.write(f"Confidence: **{top_prediction['probability'] * 100:.2f}%**")
    show_top_predictions(top_predictions)

    if top_prediction["probability"] < confidence_threshold:
        st.warning(
            "The prediction is uncertain. The image may be outside the trained "
            "classes or may not contain enough visible animal features."
        )
    else:
        st.success("The prediction passed the validated confidence rule.")

    st.subheader("Grad-CAM")
    with st.spinner("Generating explanation..."):
        class_index = class_names.index(top_prediction["class_name"])
        heatmap = make_gradcam_heatmap(model, result.processed_rgb, class_index)
        overlay = overlay_gradcam(result.processed_rgb, heatmap)
    st.image(overlay, caption="Influential image regions", use_container_width=True)
    st.caption("Grad-CAM is an explanation aid, not proof that the model understands the animal.")


def model_page():
    st.title("Model Information")
    model, class_names, confidence_threshold = load_model_bundle()
    st.write("Architecture: **EfficientNetB0 with a custom classification head**")
    st.write("Input size: **224 × 224 RGB**")
    st.write("Classes: **" + ", ".join(name.title() for name in class_names) + "**")
    st.write(f"Uncertainty threshold: **{confidence_threshold:.2f}**")
    if model is None:
        st.warning("No trained model package is installed in the models folder.")
    else:
        st.success("The trained model package is ready.")
    metrics_path = Path(__file__).resolve().parent / "models" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_columns = st.columns(2)
        metric_columns[0].metric("Test accuracy", f"{metrics.get('test_accuracy', 0) * 100:.2f}%")
        metric_columns[1].metric("Test loss", f"{metrics.get('test_loss', 0):.4f}")
    st.subheader("Limitations")
    st.markdown(
        "- The model recognizes only the ten trained classes.\n"
        "- Confidence is not perfect unknown-animal detection.\n"
        "- Performance can drop with unusual backgrounds or multiple animals.\n"
        "- Final performance must be reported from the held-out test set."
    )


st.markdown('<div class="app-header">Animal Vision System</div>', unsafe_allow_html=True)

navigation = st.radio(
    "Navigation",
    ["Home", "Process", "Predict", "Model"],
    horizontal=True,
    label_visibility="collapsed",
)

if navigation == "Home":
    home_page()
elif navigation == "Process":
    process_page()
elif navigation == "Predict":
    prediction_page()
else:
    model_page()
