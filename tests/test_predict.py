import io

import numpy as np
from PIL import Image


def test_predict_valid_image_returns_prediction(client, image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["filename"] == "scene.jpg"
    assert data["predicted_class"] == "Forest"
    assert data["final_result"] == "Forest"
    assert data["status"] == "KNOWN"

    assert abs(data["confidence"] - 0.70) < 1e-6
    assert data["confidence_percent"] == "70.00%"
    assert abs(data["p_unknown"] - 0.05) < 1e-6


def test_predict_rejects_non_image(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"not an image", "text/plain")}
    )

    assert response.status_code == 400

    assert response.json()["detail"] == ("File must be an image.")


def test_predict_rejects_corrupt_image(client):
    response = client.post(
        "/predict", files={"file": ("broken.jpg", b"not really an image", "image/jpeg")}
    )

    assert response.status_code == 400

    assert "Could not read image" in (response.json()["detail"])


def test_blank_image_is_flagged_for_review(client):
    output = io.BytesIO()

    Image.new("RGB", (64, 64), color=(100, 100, 100)).save(output, format="PNG")

    response = client.post(
        "/predict",
        files={"file": ("blank.png", output.getvalue(), "image/png")},
        params={"threshold": 0.3},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "REVIEW"
    assert data["final_result"] == "UNRECOGNIZED"
    assert data["predicted_class"] == "Unknown"
    assert data["confidence"] == 0.0


def test_high_unknown_probability_is_reviewed(client, api, image_bytes):
    api.model.probabilities = np.array(
        [[0.40, 0.10, 0.10, 0.10, 0.30]], dtype=np.float32
    )

    response = client.post(
        "/predict",
        files={"file": ("uncertain.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "REVIEW"
    assert data["final_result"] == "UNRECOGNIZED"

    # Restore the default mock prediction.
    api.model.probabilities = np.array(
        [[0.70, 0.10, 0.08, 0.07, 0.05]], dtype=np.float32
    )
