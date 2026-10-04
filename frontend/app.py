"""Premium Streamlit frontend for Satellite Scene Classification."""

import html
import io
import os
import zipfile
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Satellite Scene Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# PREMIUM UI
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #07111f;
        --panel: #0d1b2a;
        --panel-2: #102338;
        --border: rgba(148, 163, 184, 0.16);
        --text: #f8fafc;
        --muted: #94a3b8;
        --accent: #3b82f6;
        --accent-2: #60a5fa;
        --success: #22c55e;
        --warning: #f59e0b;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(59,130,246,.13), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(96,165,250,.08), transparent 24%),
            linear-gradient(180deg, #06101d 0%, #091525 45%, #07111f 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07111f 0%, #0a1626 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: var(--text);
        letter-spacing: -0.02em;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 34px 36px;
        border-radius: 24px;
        margin-bottom: 24px;
        background:
            linear-gradient(135deg, rgba(25,49,80,.96), rgba(10,24,41,.96));
        border: 1px solid rgba(96,165,250,.20);
        box-shadow: 0 20px 55px rgba(0,0,0,.22);
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 260px;
        height: 260px;
        right: -90px;
        top: -110px;
        border-radius: 50%;
        background: rgba(59,130,246,.14);
        filter: blur(6px);
    }

    .hero-kicker {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        margin-bottom: 10px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(59,130,246,.13);
        border: 1px solid rgba(96,165,250,.18);
        color: #93c5fd;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .hero-title {
        margin: 0;
        font-size: 2.55rem;
        line-height: 1.05;
        font-weight: 800;
        color: #ffffff;
    }

    .hero-subtitle {
        margin: 12px 0 0;
        max-width: 880px;
        color: #a9b8ca;
        font-size: 1.03rem;
        line-height: 1.65;
    }

    .section-head {
        margin: 4px 0 14px;
        color: #eaf2fb;
        font-weight: 750;
        font-size: 1.15rem;
    }

    .metric-card {
        padding: 18px 18px 16px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(16,35,56,.92), rgba(12,28,45,.92));
        border: 1px solid var(--border);
        box-shadow: 0 10px 35px rgba(0,0,0,.16);
    }

    .metric-label {
        color: #8fa4bb;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 700;
    }

    .metric-value {
        margin-top: 5px;
        color: white;
        font-size: 1.65rem;
        font-weight: 800;
    }

    .upload-card {
        padding: 20px;
        border-radius: 20px;
        background: rgba(13,27,42,.72);
        border: 1px solid var(--border);
    }

    .result-card {
        padding: 22px;
        border-radius: 20px;
        margin: 10px 0 16px;
        background: linear-gradient(180deg, rgba(15,31,49,.96), rgba(10,23,38,.96));
        border: 1px solid var(--border);
        box-shadow: 0 14px 38px rgba(0,0,0,.18);
    }

    .result-card.known {
        border-color: rgba(34,197,94,.32);
    }

    .result-card.review {
        border-color: rgba(245,158,11,.38);
    }

    .result-top {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 14px;
    }

    .result-name {
        color: #f8fafc;
        font-size: 1.24rem;
        font-weight: 800;
    }

    .result-meta {
        color: #93a4b8;
        font-size: .88rem;
    }

    .status-badge {
        padding: 5px 11px;
        border-radius: 999px;
        font-size: .74rem;
        font-weight: 800;
        letter-spacing: .05em;
    }

    .status-known {
        background: rgba(34,197,94,.12);
        border: 1px solid rgba(34,197,94,.24);
        color: #86efac;
    }

    .status-review {
        background: rgba(245,158,11,.12);
        border: 1px solid rgba(245,158,11,.25);
        color: #fcd34d;
    }

    .preview-tile {
        padding: 8px;
        border-radius: 16px;
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(255,255,255,.07);
    }

    .bulk-card {
        padding: 14px;
        border-radius: 18px;
        margin-bottom: 12px;
        background: rgba(12,28,45,.88);
        border: 1px solid var(--border);
    }

    .bulk-title {
        color: #f8fafc;
        font-weight: 750;
        font-size: .98rem;
        overflow-wrap: anywhere;
    }

    .bulk-meta {
        color: #94a3b8;
        font-size: .82rem;
    }

    .confidence-wrap {
        margin-top: 10px;
    }

    .confidence-track {
        width: 100%;
        height: 9px;
        border-radius: 999px;
        background: rgba(148,163,184,.12);
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #2563eb, #60a5fa, #22c55e);
    }

    .confidence-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 7px;
        color: #9fb1c5;
        font-size: .80rem;
    }

    .empty-state {
        padding: 42px 25px;
        text-align: center;
        border-radius: 20px;
        border: 1px dashed rgba(148,163,184,.22);
        background: rgba(255,255,255,.018);
        color: #91a3b7;
    }

    .stButton > button {
        min-height: 44px;
        border-radius: 12px;
        border: 1px solid rgba(96,165,250,.18);
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 750;
        box-shadow: 0 10px 24px rgba(37,99,235,.18);
    }

    .stButton > button:hover {
        border-color: rgba(147,197,253,.30);
        background: linear-gradient(135deg, #3b82f6, #2563eb);
    }

    div[data-testid="stFileUploader"] {
        border-radius: 18px;
    }

    div[data-testid="stMetric"] {
        background: transparent;
    }

    div[data-testid="stMetricLabel"] {
        color: #90a4bb;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .tab-btn {
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------
def backend_health() -> tuple[bool, float]:
    """Return backend availability and the configured review threshold."""
    try:
        health = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
        return True, float(health.get("threshold", 0.5))
    except (
        requests.exceptions.RequestException,
        ValueError,
        KeyError,
        TypeError,
    ):
        return False, 0.5


def safe_status_class(status: str) -> tuple[str, str, str]:
    """Return CSS class, badge class and icon for a prediction status."""
    if status == "KNOWN":
        return "known", "status-known", "✓"
    return "review", "status-review", "!"


def render_confidence(confidence: float) -> None:
    """Render a premium confidence progress bar."""
    percent = max(0.0, min(100.0, confidence * 100))
    st.markdown(
        f"""
        <div class="confidence-wrap">
            <div class="confidence-label">
                <span>Model confidence</span>
                <strong>{percent:.2f}%</strong>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{percent:.2f}%"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def extract_zip_images(zip_bytes: bytes) -> dict[str, bytes]:
    """Read supported images from a ZIP file into memory for UI preview only."""
    previews: dict[str, bytes] = {}
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            suffix = Path(member.filename).suffix.lower()
            if suffix not in allowed:
                continue

            # Do not write uploaded images to disk.
            previews.setdefault(Path(member.filename).name, archive.read(member))

    return previews


def render_bulk_result(
    result: dict,
    image_bytes: bytes | None,
) -> None:
    """Render one premium bulk classification result."""
    filename = str(result.get("filename", "Unknown file"))
    final_result = str(result.get("final_result", "Unknown"))
    confidence = float(result.get("confidence", 0.0))
    confidence_percent = str(
        result.get("confidence_percent", f"{confidence * 100:.2f}%")
    )
    status = str(result.get("status", "REVIEW"))

    card_class, _, icon = safe_status_class(status)
    safe_filename = html.escape(filename)
    safe_result = html.escape(final_result)

    st.markdown(
        f"""
        <div class="bulk-card {card_class}">
            <div class="bulk-title">{icon} {safe_filename}</div>
            <div class="bulk-meta">{safe_result} · {confidence_percent} · {status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preview_col, info_col = st.columns([1, 2.35], gap="large")

    with preview_col:
        if image_bytes:
            st.image(
                image_bytes,
                caption=filename,
                use_column_width=True,
            )
        else:
            st.markdown(
                '<div class="preview-tile"><div class="empty-state">Preview unavailable</div></div>',
                unsafe_allow_html=True,
            )

    with info_col:
        st.markdown(f"**{safe_result}**")
        st.caption(f"Status: {status}  ·  Confidence: {confidence_percent}")

        p_unknown = result.get("p_unknown")
        if p_unknown is not None:
            st.caption(f"P(Unknown): {float(p_unknown) * 100:.2f}%")

        render_confidence(confidence)

        if status == "REVIEW":
            st.warning("Low-confidence prediction — flagged for manual review.")
        else:
            st.success("Prediction accepted as a known scene category.")


# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ Satellite Scene Classifier")
    st.caption("EfficientNetV2-S · NWPU-RESISC45")
    st.divider()

    connected, default_threshold = backend_health()

    if connected:
        st.success("Backend connected")
    else:
        st.error("Backend not reachable")

    st.caption(f"API: {BACKEND_URL}")

    st.divider()
    st.markdown("### Review controls")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(default_threshold),
        step=0.01,
        help="Predictions below this threshold are flagged for review.",
    )

    st.caption(f"Current threshold: **{threshold:.2f}**")

# ----------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------
st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">🛰️ AI Scene Classification · Azure Ready</div>
        <h1 class="hero-title">Satellite Image Scene Classifier</h1>
        <p class="hero-subtitle">
            Upload one image or a ZIP of satellite scenes and get a structured
            classification with confidence scoring, low-confidence review flags,
            and clear per-image results.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

tab_single, tab_bulk = st.tabs(["📷  Single Image", "🗂️  Bulk ZIP Classification"])

# ----------------------------------------------------------------------
# SINGLE IMAGE
# ----------------------------------------------------------------------
with tab_single:
    st.markdown(
        '<div class="section-head">Single-image analysis</div>', unsafe_allow_html=True
    )

    upload_col, result_col = st.columns([0.95, 1.35], gap="large")

    with upload_col:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.markdown("### Upload image")
        uploaded_file = st.file_uploader(
            "JPG, JPEG, PNG, BMP or WEBP",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="single_upload",
            label_visibility="collapsed",
        )

        if uploaded_file:
            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                use_column_width=True,
            )
            classify_single = st.button(
                "🔎 Classify image",
                use_container_width=True,
            )
        else:
            classify_single = False
            st.markdown(
                """
                <div class="empty-state">
                    <strong>Ready when you are</strong><br>
                    Upload a satellite image to start the analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    with result_col:
        st.markdown(
            '<div class="section-head">Classification result</div>',
            unsafe_allow_html=True,
        )

        if not uploaded_file:
            st.markdown(
                """
                <div class="empty-state">
                    Your prediction card will appear here.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if uploaded_file and classify_single:
            with st.spinner("Running model inference..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }

                    response = requests.post(
                        f"{BACKEND_URL}/predict",
                        files=files,
                        params={"threshold": threshold},
                        timeout=60,
                    )
                    response.raise_for_status()
                    result = response.json()

                    card_class, badge_class, icon = safe_status_class(
                        result.get("status", "REVIEW")
                    )

                    safe_final = html.escape(str(result.get("final_result", "")))
                    safe_pred = html.escape(str(result.get("predicted_class", "")))
                    status = str(result.get("status", "REVIEW"))

                    st.markdown(
                        f"""
                        <div class="result-card {card_class}">
                            <div class="result-top">
                                <div>
                                    <div class="result-name">{icon} {safe_final}</div>
                                    <div class="result-meta">
                                        Predicted class · {safe_pred}
                                    </div>
                                </div>
                                <div class="status-badge {badge_class}">
                                    {status}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            '<div class="metric-card"><div class="metric-label">Predicted class</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="metric-value">{safe_pred}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            '<div class="metric-card"><div class="metric-label">Confidence</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="metric-value">{html.escape(str(result.get("confidence_percent", "N/A")))}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            '<div class="metric-card"><div class="metric-label">Status</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div class="metric-value">{status}</div></div>',
                            unsafe_allow_html=True,
                        )

                    st.markdown("### Confidence")
                    render_confidence(float(result.get("confidence", 0.0)))

                    if status == "REVIEW":
                        st.warning(
                            "This image did not confidently match a known scene category and was flagged for manual review."
                        )
                    else:
                        st.success("Prediction accepted as a known scene category.")

                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Unexpected prediction response: {exc}")

# ----------------------------------------------------------------------
# BULK ZIP
# ----------------------------------------------------------------------
with tab_bulk:
    st.markdown(
        '<div class="section-head">Bulk scene analysis</div>', unsafe_allow_html=True
    )

    zip_file = st.file_uploader(
        "Upload a ZIP containing satellite images",
        type=["zip"],
        key="bulk_upload",
        help="Supported images: JPG, JPEG, PNG, BMP, WEBP.",
    )

    if not zip_file:
        st.markdown(
            """
            <div class="empty-state">
                <strong>Bulk workspace</strong><br>
                Upload one ZIP file and every image will be shown with its
                individual prediction, confidence and review status.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if zip_file:
        zip_bytes = zip_file.getvalue()
        preview_images = {}

        st.markdown("### Upload summary")
        s1, s2 = st.columns(2)

        with s1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">ZIP file</div>
                    <div class="metric-value" style="font-size:1.05rem;">
                        {html.escape(zip_file.name)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with s2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Images detected</div>
                    <div class="metric-value">{len(preview_images)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        run_bulk = st.button(
            "🚀 Classify all images",
            use_container_width=True,
        )

        if run_bulk:
            with st.spinner("Processing ZIP and classifying images..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/predict/bulk",
                        files={
                            "file": (
                                zip_file.name,
                                zip_bytes,
                                "application/zip",
                            )
                        },
                        params={"threshold": threshold},
                        timeout=180,
                    )
                    response.raise_for_status()
                    bulk_result = response.json()

                    results = bulk_result.get("results", [])

                    total = int(bulk_result.get("total", len(results)))
                    known = int(bulk_result.get("known", 0))
                    review = int(bulk_result.get("review", 0))

                    st.markdown("### Batch overview")
                    a, b, c = st.columns(3)

                    with a:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Total images</div>
                                <div class="metric-value">{total}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with b:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Accepted</div>
                                <div class="metric-value">{known}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with c:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-label">Needs review</div>
                                <div class="metric-value">{review}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if total:
                        st.progress(
                            min(1.0, known / total),
                            text=f"{known} of {total} images accepted",
                        )

                    st.markdown("### Image-by-image results")

                    try:
                        preview_images = extract_zip_images(zip_bytes)
                    except (OSError, ValueError, zipfile.BadZipFile, KeyError):
                        preview_images = {}

                    # Every returned image gets its own visual result block.
                    for item in results:
                        filename = str(item.get("filename", ""))
                        image_bytes = preview_images.get(Path(filename).name)
                        render_bulk_result(item, image_bytes)

                    result_df = pd.DataFrame(results)

                    if not result_df.empty:
                        display_df = result_df.rename(
                            columns={
                                "filename": "Image",
                                "final_result": "Result",
                                "confidence_percent": "Confidence",
                                "status": "Status",
                            }
                        )

                        st.markdown("### Results table")
                        st.dataframe(
                            display_df[["Image", "Result", "Confidence", "Status"]],
                            use_container_width=True,
                            hide_index=True,
                        )

                        csv_data = display_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download CSV report",
                            data=csv_data,
                            file_name="classification_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                except requests.exceptions.RequestException as exc:
                    st.error(f"Could not reach backend: {exc}")
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Unexpected bulk response: {exc}")

st.divider()
st.caption(
    "Satellite Image Scene Classification · EfficientNetV2-S · FastAPI + Streamlit · Azure Container Apps"
)
