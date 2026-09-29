"""
FastAPI backend for the Satellite Scene Classification project.

Endpoints (matches BRD FR-8, FR-9, FR-10, FR-11, FR-12, FR-13):
  GET  /health              -> simple check that the model is loaded
  POST /predict              -> single image upload, returns class + confidence
  POST /predict/bulk         -> zip file upload, returns a results table
  GET  /config                -> current threshold + class list (for the UI to show)

Run with:
    uvicorn main:app --reload --port 8000
"""

import io
import json
import zipfile
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ----------------------------------------------------------------------
# CONFIG — only thing you need to edit
# ----------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "satellite_v5_final.keras"      # <-- your downloaded .keras file
CONFIG_PATH = MODEL_DIR / "config.json"                   # <-- class_names + unk_threshold

IMG_SIZE = 224

# ----------------------------------------------------------------------
# LOAD MODEL + CONFIG ONCE AT STARTUP
# ----------------------------------------------------------------------
app = FastAPI(title="Satellite Scene Classification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # for local dev; restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Download it from your Kaggle notebook's /kaggle/working/.../models/ folder "
        f"and place it here."
    )
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}.")

model = keras.models.load_model(MODEL_PATH, compile=False)   # compile=False: we only need inference
config = json.loads(CONFIG_PATH.read_text())

CLASS_NAMES: List[str] = config["class_names"]        # e.g. ["Forest","SeaLake","Desert","Cloudy","Unknown"]
UNK_THRESHOLD: float = config.get("unk_threshold", config.get("conf_threshold", 0.5))
UNKNOWN_ID = CLASS_NAMES.index("Unknown") if "Unknown" in CLASS_NAMES else None
KNOWN_CLASS_NAMES = [c for c in CLASS_NAMES if c != "Unknown"]

print(f"[startup] Model loaded from {MODEL_PATH}")
print(f"[startup] Classes: {CLASS_NAMES} | threshold: {UNK_THRESHOLD}")


# ----------------------------------------------------------------------
# RESPONSE SCHEMAS (FR-10: confidence score on every response)
# ----------------------------------------------------------------------
class PredictionResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float          # 0-1
    confidence_percent: str    # "93.60%" for display convenience
    p_unknown: float
    final_result: str          # class name OR "UNRECOGNIZED"
    status: str                # "KNOWN" or "REVIEW"  (FR-12)


class BulkPredictionResponse(BaseModel):
    total: int
    known: int
    review: int
    results: List[PredictionResult]


# ----------------------------------------------------------------------
# CORE PREDICTION LOGIC (shared by single + bulk endpoints)
# ----------------------------------------------------------------------
def load_and_prepare(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def is_blank(arr: np.ndarray, std_threshold: float = 1.0) -> bool:
    """Catch literally blank/solid-color images before running the model."""
    return float(np.std(arr[0])) < std_threshold


def predict_array(arr: np.ndarray, filename: str, threshold: float = None) -> PredictionResult:
    T = UNK_THRESHOLD if threshold is None else threshold

    if is_blank(arr):
        return PredictionResult(
            filename=filename, predicted_class="Unknown", confidence=0.0,
            confidence_percent="0.00%", p_unknown=1.0,
            final_result="UNRECOGNIZED", status="REVIEW",
        )

    # 4-view test-time augmentation, one batched call
    views = np.concatenate([
        arr,
        arr[:, :, ::-1, :],
        arr[:, ::-1, :, :],
        np.rot90(arr[0], k=2)[None, ...],
    ], axis=0)

    probs = model.predict(views, verbose=0).mean(axis=0)

    if UNKNOWN_ID is not None:
        top = int(probs.argmax())
        p_unknown = float(probs[UNKNOWN_ID])
        known_probs = np.delete(probs, UNKNOWN_ID)
        best_known_idx = int(known_probs.argmax())
        accept = (top != UNKNOWN_ID) and (p_unknown < T)
    else:
        # model has no explicit Unknown class -> fall back to confidence threshold
        top = int(probs.argmax())
        p_unknown = 0.0
        best_known_idx = top
        accept = float(probs[top]) >= T

    predicted_class = KNOWN_CLASS_NAMES[best_known_idx]
    confidence = float(probs[CLASS_NAMES.index(predicted_class)])

    return PredictionResult(
        filename=filename,
        predicted_class=predicted_class,
        confidence=confidence,
        confidence_percent=f"{confidence * 100:.2f}%",
        p_unknown=p_unknown,
        final_result=predicted_class if accept else "UNRECOGNIZED",
        status="KNOWN" if accept else "REVIEW",
    )


# ----------------------------------------------------------------------
# ENDPOINTS
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "classes": CLASS_NAMES, "threshold": UNK_THRESHOLD}


@app.get("/config")
def get_config():
    """So the frontend can show the threshold and let the user tweak it (FR-13)."""
    return {"class_names": KNOWN_CLASS_NAMES, "default_threshold": UNK_THRESHOLD}


@app.post("/predict", response_model=PredictionResult)
async def predict_single(file: UploadFile = File(...), threshold: float = None):
    """FR-8: single image upload -> predicted class + confidence."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")
    content = await file.read()
    try:
        arr = load_and_prepare(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")
    return predict_array(arr, file.filename, threshold)


@app.post("/predict/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(file: UploadFile = File(...), threshold: float = None):
    """FR-9 + FR-11: zip upload -> results table for every image inside."""
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a .zip archive.")

    content = await file.read()
    results: List[PredictionResult] = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [n for n in zf.namelist() if Path(n).suffix.lower() in valid_ext and not n.startswith("__MACOSX")]
            if not names:
                raise HTTPException(status_code=400, detail="No images found inside the zip.")
            for name in names:
                try:
                    img_bytes = zf.read(name)
                    arr = load_and_prepare(img_bytes)
                    results.append(predict_array(arr, Path(name).name, threshold))
                except Exception as e:
                    results.append(PredictionResult(
                        filename=Path(name).name, predicted_class="ERROR", confidence=0.0,
                        confidence_percent="0.00%", p_unknown=0.0,
                        final_result=f"ERROR: {e}", status="REVIEW",
                    ))
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file.")

    known = sum(1 for r in results if r.status == "KNOWN")
    return BulkPredictionResponse(
        total=len(results), known=known, review=len(results) - known, results=results
    )
