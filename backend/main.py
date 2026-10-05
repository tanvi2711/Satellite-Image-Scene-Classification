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

import hashlib
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
from sqlalchemy.exc import SQLAlchemyError
from tensorflow import keras

from backend.database import PredictionLog, SessionLocal

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "satellite_v5_final.keras"
CONFIG_PATH = MODEL_DIR / "config.json"

IMG_SIZE = 224
BULK_BATCH_SIZE = 32

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger("satellite_classifier")


def calculate_image_hash(image_bytes: bytes) -> str:
    """Create a unique SHA-256 fingerprint for the uploaded image."""
    return hashlib.sha256(image_bytes).hexdigest()


def log_prediction(
    *,
    request_id: str,
    mode: str,
    filename: str,
    image_hash: str,
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

    db = SessionLocal()

    try:
        db_log = PredictionLog(
            event="prediction",
            request_id=request_id,
            mode=mode,
            filename=filename,
            image_hash=image_hash,
            predicted_class=predicted_class,
            confidence=round(confidence, 6),
            confidence_percent=round(confidence * 100, 2),
            p_unknown=round(p_unknown, 6),
            status=status,
            final_result=final_result,
            threshold=round(threshold, 6),
            processing_time_ms=round(processing_time_ms, 2),
            cache_hit=False,
        )

        db.add(db_log)
        db.commit()

    except SQLAlchemyError as exc:
        db.rollback()
        logger.error(
            json.dumps(
                {
                    "event": "database_error",
                    "request_id": request_id,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )

    finally:
        db.close()


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
print(f"[startup] Bulk inference batch size: {BULK_BATCH_SIZE}")


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


def make_tta_views(arrays: list[np.ndarray]) -> np.ndarray:
    """Create four test-time views for each image and return one model batch."""
    batches: list[np.ndarray] = []

    for arr in arrays:
        batches.extend(
            [
                arr,
                arr[:, :, ::-1, :],
                arr[:, ::-1, :, :],
                np.rot90(arr[0], k=2)[None, ...],
            ]
        )

    return np.concatenate(batches, axis=0)


def result_from_probabilities(
    probs: np.ndarray,
    filename: str,
    threshold: float,
) -> PredictionResult:
    """Convert one averaged probability vector into the API result schema."""
    if UNKNOWN_ID is not None:
        top = int(probs.argmax())
        p_unknown = float(probs[UNKNOWN_ID])
        known_probs = np.delete(probs, UNKNOWN_ID)
        best_known_idx = int(known_probs.argmax())
        accept = (top != UNKNOWN_ID) and (p_unknown < threshold)
    else:
        top = int(probs.argmax())
        p_unknown = 0.0
        best_known_idx = top
        accept = float(probs[top]) >= threshold

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


def predict_array(
    arr: np.ndarray,
    filename: str,
    threshold: float | None = None,
) -> PredictionResult:
    """Run model inference for one image using four-view test-time augmentation."""
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

    views = make_tta_views([arr])
    probs = model.predict(views, verbose=0).mean(axis=0)
    return result_from_probabilities(probs, filename, selected_threshold)


def predict_batch(
    arrays: list[np.ndarray],
    filenames: list[str],
    threshold: float,
) -> list[PredictionResult]:
    """Predict a batch of non-blank images with one TensorFlow model call."""
    if not arrays:
        return []

    views = make_tta_views(arrays)
    probabilities = model.predict(views, verbose=0)

    # Four views are stored consecutively for each source image.
    probabilities = probabilities.reshape(
        len(arrays),
        4,
        len(CLASS_NAMES),
    ).mean(axis=1)

    return [
        result_from_probabilities(probabilities[index], filenames[index], threshold)
        for index in range(len(arrays))
    ]


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
    image_hash = calculate_image_hash(content)

    db = SessionLocal()

    try:
        cached_log = (
            db.query(PredictionLog)
            .filter(
                PredictionLog.image_hash == image_hash,
                PredictionLog.event == "prediction",
                PredictionLog.threshold
                == (UNK_THRESHOLD if threshold is None else threshold),
            )
            .order_by(PredictionLog.id.desc())
            .first()
        )

        if cached_log:
            processing_time_ms = (time.perf_counter() - start) * 1000

            logger.info(
                json.dumps(
                    {
                        "event": "cache_hit",
                        "request_id": request_id,
                        "mode": "single",
                        "filename": file.filename,
                        "image_hash": image_hash,
                        "predicted_class": cached_log.predicted_class,
                        "confidence": cached_log.confidence,
                        "final_result": cached_log.final_result,
                        "processing_time_ms": round(processing_time_ms, 2),
                    }
                )
            )

            return {
                "success": True,
                "request_id": request_id,
                "filename": file.filename,
                "predicted_class": cached_log.predicted_class,
                "confidence": cached_log.confidence,
                "confidence_percent": f"{cached_log.confidence_percent:.2f}%",
                "p_unknown": cached_log.p_unknown,
                "status": cached_log.status,
                "final_result": cached_log.final_result,
                "threshold": cached_log.threshold,
                "processing_time_ms": round(processing_time_ms, 2),
                "cache_hit": True,
            }

    finally:
        db.close()

    try:
        arr = load_and_prepare(content)
        result = predict_array(arr, filename, threshold)

        elapsed_ms = (time.perf_counter() - start) * 1000

        log_prediction(
            request_id=request_id,
            mode="single",
            filename=filename,
            image_hash=image_hash,
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
    """Classify every supported image in a ZIP using batched inference."""
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
    selected_threshold = UNK_THRESHOLD if threshold is None else threshold

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

            # Process images in bounded batches to improve throughput without
            # putting an unnecessarily large number of images into RAM at once.
            for batch_start in range(0, len(names), BULK_BATCH_SIZE):
                batch_names = names[batch_start : batch_start + BULK_BATCH_SIZE]
                batch_arrays: list[np.ndarray] = []
                batch_filenames: list[str] = []
                batch_hashes: list[str] = []
                batch_results: list[PredictionResult | None] = [None] * len(batch_names)

                for index, name in enumerate(batch_names):
                    image_start = time.perf_counter()
                    image_name = Path(name).name

                    try:
                        img_bytes = zf.read(name)
                        image_hash = calculate_image_hash(img_bytes)

                        db = SessionLocal()

                        try:
                            cached_log = (
                                db.query(PredictionLog)
                                .filter(
                                    PredictionLog.image_hash == image_hash,
                                    PredictionLog.event == "prediction",
                                    PredictionLog.threshold == selected_threshold,
                                )
                                .order_by(PredictionLog.id.desc())
                                .first()
                            )
                        finally:
                            db.close()

                        if cached_log:
                            cached_result = PredictionResult(
                                filename=image_name,
                                predicted_class=cached_log.predicted_class,
                                confidence=cached_log.confidence,
                                confidence_percent=cached_log.confidence_percent,
                                p_unknown=cached_log.p_unknown,
                                final_result=cached_log.final_result,
                                status=cached_log.status,
                            )

                            batch_results[index] = cached_result

                            logger.info(
                                json.dumps(
                                    {
                                        "event": "cache_hit",
                                        "request_id": request_id,
                                        "mode": "bulk",
                                        "filename": image_name,
                                        "image_hash": image_hash,
                                        "predicted_class": cached_log.predicted_class,
                                        "confidence": cached_log.confidence,
                                        "final_result": cached_log.final_result,
                                    },
                                    ensure_ascii=False,
                                )
                            )

                            continue

                        arr = load_and_prepare(img_bytes)

                        if is_blank(arr):
                            blank_result = PredictionResult(
                                filename=image_name,
                                predicted_class="Unknown",
                                confidence=0.0,
                                confidence_percent="0.00%",
                                p_unknown=1.0,
                                final_result="UNRECOGNIZED",
                                status="REVIEW",
                            )
                            batch_results[index] = blank_result
                            log_prediction(
                                request_id=request_id,
                                mode="bulk",
                                filename=image_name,
                                image_hash=image_hash,
                                predicted_class=blank_result.predicted_class,
                                confidence=blank_result.confidence,
                                p_unknown=blank_result.p_unknown,
                                final_result=blank_result.final_result,
                                status=blank_result.status,
                                threshold=selected_threshold,
                                processing_time_ms=(time.perf_counter() - image_start)
                                * 1000,
                            )
                            continue

                        batch_arrays.append(arr)
                        batch_filenames.append(image_name)
                        batch_hashes.append(image_hash)

                    except (OSError, ValueError, RuntimeError, KeyError) as exc:
                        image_elapsed_ms = (time.perf_counter() - image_start) * 1000
                        error_message = str(exc)

                        error_result = PredictionResult(
                            filename=image_name,
                            predicted_class="ERROR",
                            confidence=0.0,
                            confidence_percent="0.00%",
                            p_unknown=0.0,
                            final_result=f"ERROR: {error_message}",
                            status="REVIEW",
                        )
                        batch_results[index] = error_result
                        log_prediction_error(
                            request_id=request_id,
                            mode="bulk",
                            filename=image_name,
                            error=error_message,
                            processing_time_ms=image_elapsed_ms,
                        )

                if batch_arrays:
                    inference_start = time.perf_counter()
                    predicted_results = predict_batch(
                        batch_arrays,
                        batch_filenames,
                        selected_threshold,
                    )
                    batch_elapsed_ms = (time.perf_counter() - inference_start) * 1000

                    average_image_ms = (
                        batch_elapsed_ms / len(predicted_results)
                        if predicted_results
                        else 0.0
                    )

                    predicted_index = 0
                    for index, item in enumerate(batch_results):
                        if item is not None:
                            continue
                        result = predicted_results[predicted_index]
                        image_hash_for_result = batch_hashes[predicted_index]

                        batch_results[index] = result
                        predicted_index += 1

                        log_prediction(
                            request_id=request_id,
                            mode="bulk",
                            filename=result.filename,
                            image_hash=image_hash_for_result,
                            predicted_class=result.predicted_class,
                            confidence=result.confidence,
                            p_unknown=result.p_unknown,
                            final_result=result.final_result,
                            status=result.status,
                            threshold=selected_threshold,
                            processing_time_ms=average_image_ms,
                        )

                results.extend(result for result in batch_results if result is not None)

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
                "batch_size": BULK_BATCH_SIZE,
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
