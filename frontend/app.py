"""
Streamlit frontend for the Satellite Scene Classification project.

Run with:
    streamlit run app.py

Make sure the FastAPI backend is already running at BACKEND_URL below.
"""

import io
import requests
import pandas as pd
import streamlit as st
import os

# ----------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="Satellite Scene Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PREMIUM STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0b1220 0%, #0f1b2d 100%);
        color: #e8edf5;
    }
    section[data-testid="stSidebar"] {
        background: #0c1524;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    h1, h2, h3 { color: #f5f7fb; font-weight: 700; }
    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(135deg, #14243f 0%, #0e1a30 100%);
        border: 1px solid rgba(120,170,255,0.15);
        margin-bottom: 28px;
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 2.1rem; }
    .hero p { color: #9fb0c9; font-size: 1.02rem; margin: 0; }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background: rgba(88,166,255,0.15); color: #7db4ff; font-size: 0.78rem;
        font-weight: 600; letter-spacing: 0.03em; margin-right: 8px;
    }
    .result-card {
        padding: 20px 24px; border-radius: 14px; margin-bottom: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .result-known { background: rgba(46,160,67,0.10); border-color: rgba(46,160,67,0.35); }
    .result-review { background: rgba(210,153,34,0.10); border-color: rgba(210,153,34,0.35); }
    .result-title { font-size: 1.3rem; font-weight: 700; margin-bottom: 4px; }
    .result-sub { color: #9fb0c9; font-size: 0.92rem; }
    .conf-bar-bg { background: rgba(255,255,255,0.08); border-radius: 8px; height: 10px; margin-top: 10px; }
    .conf-bar-fill { height: 10px; border-radius: 8px; background: linear-gradient(90deg,#58a6ff,#7ee787); }
    div[data-testid="stMetricValue"] { color: #e8edf5; }
    .stButton>button {
        background: linear-gradient(135deg,#2563eb,#1d4ed8); color: white; border: none;
        border-radius: 10px; padding: 0.55rem 1.4rem; font-weight: 600;
    }
    .stButton>button:hover { background: linear-gradient(135deg,#1d4ed8,#1e40af); }
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 🛰️ Satellite Scene Classifier")
    st.caption("EfficientNetV2-S · NWPU-RESISC45")
    st.divider()

    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        st.success("Backend connected")
        st.caption(f"Classes: {', '.join(health['classes'])}")
        default_threshold = health.get("threshold", 0.5)
    except Exception:
        st.error("Backend not reachable. Start it with:\n`uvicorn main:app --reload`")
        default_threshold = 0.5

    st.divider()
    st.markdown("**Review threshold**")
    threshold = st.slider(
        "Lower = stricter (more images flagged for review)",
        min_value=0.05, max_value=0.95, value=float(default_threshold), step=0.01,
    )
    # st.caption("Adjustable per FR-13 — not hard-coded.")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <span class="badge">SCENE CLASSIFICATION</span><span class="badge">AI-POWERED</span>
    <h1>Satellite Image Scene Classifier</h1>
    <p>Upload a satellite image chip and get an instant scene classification — Forest, Sea/Lake,
    Desert or Cloudy — with a confidence score. Anything that doesn't match a known scene is
    automatically flagged for review instead of a forced, unreliable guess.</p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_bulk = st.tabs(["📷 Single Image", "🗂️ Bulk ZIP Upload"])

# ======================================================================
# TAB 1 — SINGLE IMAGE
# ======================================================================
with tab_single:
    col_upload, col_result = st.columns([1, 1.3], gap="large")

    with col_upload:
        st.markdown("#### Upload an image")
        uploaded_file = st.file_uploader(
            "Drag and drop a satellite image chip", type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="single_upload",
        )
        if uploaded_file:
            st.image(
    uploaded_file,
    use_column_width=True,
    caption=uploaded_file.name
)
            run = st.button("🔍 Classify Image", use_container_width=True)
        else:
            run = False
            st.info("Choose an image to get started.")

    with col_result:
        st.markdown("#### Result")
        if uploaded_file and run:
            with st.spinner("Running model..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    r = requests.post(f"{BACKEND_URL}/predict", files=files,
                                       params={"threshold": threshold}, timeout=30)
                    r.raise_for_status()
                    res = r.json()

                    css_class = "result-known" if res["status"] == "KNOWN" else "result-review"
                    icon = "✅" if res["status"] == "KNOWN" else "⚠️"

                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <div class="result-title">{icon} {res['final_result']}</div>
                        <div class="result-sub">Confidence: {res['confidence_percent']} &nbsp;|&nbsp;
                        Status: {res['status']} &nbsp;|&nbsp; P(Unknown): {res['p_unknown']*100:.1f}%</div>
                        <div class="conf-bar-bg">
                            <div class="conf-bar-fill" style="width:{res['confidence']*100:.1f}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Predicted Class", res["predicted_class"])
                    c2.metric("Confidence", res["confidence_percent"])
                    c3.metric("Status", res["status"])

                    if res["status"] == "REVIEW":
                        st.warning("This image doesn't confidently match a known scene category and has been flagged for manual review.")

                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach backend: {e}")
        else:
            st.markdown("Upload an image and click **Classify Image** to see results here.")

# ======================================================================
# TAB 2 — BULK ZIP UPLOAD
# ======================================================================
with tab_bulk:
    st.markdown("#### Upload a ZIP of images")
    zip_file = st.file_uploader("Drag and drop a .zip file", type=["zip"], key="bulk_upload")

    if zip_file:
        run_bulk = st.button("🔍 Classify All Images", use_container_width=True)
    else:
        run_bulk = False
        st.info("Choose a .zip file containing images to classify them all at once.")

    if zip_file and run_bulk:
        with st.spinner("Processing zip file..."):
            try:
                files = {"file": (zip_file.name, zip_file.getvalue(), "application/zip")}
                r = requests.post(f"{BACKEND_URL}/predict/bulk", files=files,
                                   params={"threshold": threshold}, timeout=120)
                r.raise_for_status()
                res = r.json()

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Images", res["total"])
                c2.metric("Known", res["known"])
                c3.metric("Needs Review", res["review"])

                df = pd.DataFrame(res["results"])
                df = df.rename(columns={
                    "filename": "Image", "final_result": "Result",
                    "confidence_percent": "Confidence", "status": "Status",
                })

                def highlight_status(row):
                    color = "background-color: rgba(46,160,67,0.15)" if row.Status == "KNOWN" else "background-color: rgba(210,153,34,0.15)"
                    return [color] * len(row)

                st.dataframe(
    df[["Image", "Result", "Confidence", "Status"]].style.apply(
        highlight_status, axis=1
    ),
    use_container_width=True,
    hide_index=True,
)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download results as CSV", data=csv,
                                   file_name="classification_results.csv", mime="text/csv")

            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")

st.divider()
st.caption("Satellite Image Scene Classification · EfficientNetV2-S · Built for the Azure Internship Project")
