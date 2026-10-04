"""
FastAPI backend for the Satellite Scene Classification project.

Endpoints:
  GET  /health
  POST /predict
  POST /predict/bulk
  GET  /config

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import io
import json
import logging
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from tensorflow import keras

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "satellite_v5_final.keras"
CONFIG_PATH = MODEL_DIR / "config.json"

IMG_SIZE = 224

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
# Logs go to stdout/stderr, so Azure Container Apps can collect them
# through the existing Log Analytics configuration.
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("satellite_classifier")


def log_prediction(
    *,
    request_id: str,
    mode: str,
    filename: str,
    predicted_class: str,
    confidence: float,
    p_unknown: float,
    final_result: str,
    status: str,
    threshold: float,
    processing_time_ms: float,
) -> None:
    """Write structured prediction metadata without storing image data."""
    log_record = {
        "event": "prediction",
        "request_id": request_id,
        "mode": mode,
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "confidence_percent": round(confidence * 100, 2),
        "p_unknown": round(p_unknown, 6),
        "status": status,
        "final_result": final_result,
        "threshold": round(threshold, 6),
        "processing_time_ms": round(processing_time_ms, 2),
    }
    logger.info(json.dumps(log_record, ensure_ascii=False))


def log_prediction_error(
    *,
    request_id: str,
    mode: str,
    filename: str,
    error: str,
    processing_time_ms: float,
) -> None:
    """Write structured prediction error metadata without storing image data."""
    log_record = {
        "event": "prediction_error",
        "request_id": request_id,
        "mode": mode,
        "filename": filename,
        "error": error,
        "processing_time_ms": round(processing_time_ms, 2),
    }
    logger.error(json.dumps(log_record, ensure_ascii=False))


# ----------------------------------------------------------------------
# LOAD MODEL + CONFIG ONCE AT STARTUP
# ----------------------------------------------------------------------
app = FastAPI(title="Satellite Scene Classification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        "Download it from the Kaggle notebook and place it in the model folder."
    )

if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"config.json not found at {CONFIG_PATH}.")

model = keras.models.load_model(MODEL_PATH, compile=False)
config = json.loads(CONFIG_PATH.read_text())

CLASS_NAMES: list[str] = config["class_names"]
UNK_THRESHOLD: float = config.get(
    "unk_threshold",
    config.get("conf_threshold", 0.5),
)
UNKNOWN_ID = CLASS_NAMES.index("Unknown") if "Unknown" in CLASS_NAMES else None
KNOWN_CLASS_NAMES = [c for c in CLASS_NAMES if c != "Unknown"]

print(f"[startup] Model loaded from {MODEL_PATH}")
print(f"[startup] Classes: {CLASS_NAMES} | threshold: {UNK_THRESHOLD}")


# ----------------------------------------------------------------------
# RESPONSE SCHEMAS
# ----------------------------------------------------------------------
class PredictionResult(BaseModel):
    filename: str
    predicted_class: str
    confidence: float
    confidence_percent: str
    p_unknown: float
    final_result: str
    status: str


class BulkPredictionResponse(BaseModel):
    total: int
    known: int
    review: int
    results: list[PredictionResult]


# ----------------------------------------------------------------------
# CORE PREDICTION LOGIC
# ----------------------------------------------------------------------
def load_and_prepare(image_bytes: bytes) -> np.ndarray:
    """Load an image, convert it to RGB, resize it and create a model batch."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR,
    )
    arr = np.asarray(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def is_blank(arr: np.ndarray, std_threshold: float = 1.0) -> bool:
    """Catch literally blank/solid-color images before running the model."""
    return float(np.std(arr[0])) < std_threshold


def predict_array(
    arr: np.ndarray,
    filename: str,
    threshold: float | None = None,
) -> PredictionResult:
    """Run model inference and apply the Unknown/review decision logic."""
    selected_threshold = UNK_THRESHOLD if threshold is None else threshold

    if is_blank(arr):
        return PredictionResult(
            filename=filename,
            predicted_class="Unknown",
            confidence=0.0,
            confidence_percent="0.00%",
            p_unknown=1.0,
            final_result="UNRECOGNIZED",
            status="REVIEW",
        )

    # Four-view test-time augmentation in one batched model call.
    views = np.concatenate(
        [
            arr,
            arr[:, :, ::-1, :],
            arr[:, ::-1, :, :],
            np.rot90(arr[0], k=2)[None, ...],
        ],
        axis=0,
    )

    probs = model.predict(views, verbose=0).mean(axis=0)

    if UNKNOWN_ID is not None:
        top = int(probs.argmax())
        p_unknown = float(probs[UNKNOWN_ID])
        known_probs = np.delete(probs, UNKNOWN_ID)
        best_known_idx = int(known_probs.argmax())
        accept = (top != UNKNOWN_ID) and (p_unknown < selected_threshold)
    else:
        top = int(probs.argmax())
        p_unknown = 0.0
        best_known_idx = top
        accept = float(probs[top]) >= selected_threshold

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
    """Return model availability, class names and the default threshold."""
    return {
        "status": "ok",
        "model_loaded": True,
        "classes": CLASS_NAMES,
        "threshold": UNK_THRESHOLD,
    }


@app.get("/config")
def get_config():
    """Return the known classes and default review threshold."""
    return {
        "class_names": KNOWN_CLASS_NAMES,
        "default_threshold": UNK_THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResult)
async def predict_single(
    file: UploadFile = File(...),  # noqa: B008
    threshold: float | None = None,
) -> PredictionResult:
    """Classify one uploaded image and log prediction metadata."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    content_type = file.content_type or ""
    filename = file.filename or "uploaded_image"

    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    content = await file.read()

    try:
        arr = load_and_prepare(content)
        result = predict_array(arr, filename, threshold)

        elapsed_ms = (time.perf_counter() - start) * 1000

        log_prediction(
            request_id=request_id,
            mode="single",
            filename=filename,
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            p_unknown=result.p_unknown,
            final_result=result.final_result,
            status=result.status,
            threshold=UNK_THRESHOLD if threshold is None else threshold,
            processing_time_ms=elapsed_ms,
        )

        return result

    except (OSError, ValueError, RuntimeError) as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_prediction_error(
            request_id=request_id,
            mode="single",
            filename=filename,
            error=str(exc),
            processing_time_ms=elapsed_ms,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        )


@app.post("/predict/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(
    file: UploadFile = File(...),  # noqa: B008
    threshold: float | None = None,
) -> BulkPredictionResponse:
    """Classify every supported image in a ZIP and log each prediction."""
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    filename = file.filename or ""

    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="File must be a .zip archive.",
        )

    content = await file.read()
    results: list[PredictionResult] = []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = [
                name
                for name in zf.namelist()
                if Path(name).suffix.lower() in valid_ext
                and not name.startswith("__MACOSX")
            ]

            if not names:
                raise HTTPException(
                    status_code=400,
                    detail="No images found inside the zip.",
                )

            for name in names:
                image_start = time.perf_counter()
                image_name = Path(name).name

                try:
                    img_bytes = zf.read(name)
                    arr = load_and_prepare(img_bytes)
                    result = predict_array(arr, image_name, threshold)
                    results.append(result)

                    image_elapsed_ms = (time.perf_counter() - image_start) * 1000

                    log_prediction(
                        request_id=request_id,
                        mode="bulk",
                        filename=image_name,
                        predicted_class=result.predicted_class,
                        confidence=result.confidence,
                        p_unknown=result.p_unknown,
                        final_result=result.final_result,
                        status=result.status,
                        threshold=(UNK_THRESHOLD if threshold is None else threshold),
                        processing_time_ms=image_elapsed_ms,
                    )

                except (OSError, ValueError, RuntimeError, KeyError) as exc:
                    image_elapsed_ms = (time.perf_counter() - image_start) * 1000

                    error_message = str(exc)

                    results.append(
                        PredictionResult(
                            filename=image_name,
                            predicted_class="ERROR",
                            confidence=0.0,
                            confidence_percent="0.00%",
                            p_unknown=0.0,
                            final_result=f"ERROR: {error_message}",
                            status="REVIEW",
                        )
                    )

                    log_prediction_error(
                        request_id=request_id,
                        mode="bulk",
                        filename=image_name,
                        error=error_message,
                        processing_time_ms=image_elapsed_ms,
                    )

    except zipfile.BadZipFile:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_prediction_error(
            request_id=request_id,
            mode="bulk",
            filename=filename,
            error="Invalid zip file.",
            processing_time_ms=elapsed_ms,
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid zip file.",
        )

    known = sum(1 for result in results if result.status == "KNOWN")

    logger.info(
        json.dumps(
            {
                "event": "bulk_complete",
                "request_id": request_id,
                "filename": filename,
                "total": len(results),
                "known": known,
                "review": len(results) - known,
                "processing_time_ms": round(
                    (time.perf_counter() - start) * 1000,
                    2,
                ),
            },
            ensure_ascii=False,
        )
    )

    return BulkPredictionResponse(
        total=len(results),
        known=known,
        review=len(results) - known,
        results=results,
    )
