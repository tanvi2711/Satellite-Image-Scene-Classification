import importlib
import io
import json
import sys
from unittest.mock import patch

import numpy as np
import pytest


# Mock model for testing
class FakeModel:
    def __init__(self):
        self.predict_calls = 0
        self.last_batch_size = 0
        self.probabilities = np.array(
            [[0.70, 0.10, 0.08, 0.07, 0.05]], dtype=np.float32
        )

    def predict(self, images, verbose=0):
        self.predict_calls += 1
        self.last_batch_size = len(images)
        return np.repeat(self.probabilities, repeats=len(images), axis=0)


@pytest.fixture(scope="session")
def api():
    fake_model = FakeModel()

    # Import the backend with the real model loading disabled.
    with (
        patch("pathlib.Path.exists", return_value=True),
        patch(
            "pathlib.Path.read_text",
            return_value=json.dumps(
                {
                    "class_names": ["Forest", "SeaLake", "Desert", "Cloudy", "Unknown"],
                    "unk_threshold": 0.3,
                }
            ),
        ),
        patch("tensorflow.keras.models.load_model", return_value=fake_model),
    ):
        sys.modules.pop("backend.main", None)
        module = importlib.import_module("backend.main")

    module.model = fake_model

    return module


@pytest.fixture
def client(api):
    api.model.predict_calls = 0
    api.model.last_batch_size = 0

    from fastapi.testclient import TestClient

    from backend.database import PredictionLog, SessionLocal

    db = SessionLocal()

    try:
        db.query(PredictionLog).delete()
        db.commit()
    finally:
        db.close()

    return TestClient(api.app)


@pytest.fixture
def image_bytes():
    from PIL import Image

    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:32, :, 0] = 255
    pixels[32:, :, 1] = 180

    output = io.BytesIO()

    Image.fromarray(pixels).save(output, format="JPEG")

    return output.getvalue()
