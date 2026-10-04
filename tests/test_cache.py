"""
Tests for the duplicate-image prediction cache.

Verifies:
A. First upload -> cache MISS -> model inference occurs.
B. Same exact image -> cache HIT -> model inference does NOT occur.
C. Same image with different filename -> cache HIT.
D. Slightly modified image -> cache MISS -> model inference occurs.
E. Different model version -> cache MISS.
F. Bulk ZIP with duplicate images -> inference runs only for unique images.
G. Dynamic threshold behavior works on cached predictions.
H. Blank images are not cached.
I. Direct unit tests for cache backends (In-Memory, SQLite, and Azure Table mapping).
"""

import io
import json
import zipfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from backend.cache import (
    AzureTablePredictionCache,
    InMemoryPredictionCache,
    SQLitePredictionCache,
    compute_image_hash,
)


@pytest.fixture
def modified_image_bytes(image_bytes):
    """Generate an image that differs from image_bytes by altering pixel values."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pixels = np.array(img)
    # Invert the pixel values to guarantee a completely different hash
    pixels = 255 - pixels
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="JPEG")
    return out.getvalue()


@pytest.fixture
def secondary_image_bytes():
    """Generate a second distinct valid test image."""
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, :32, 2] = 200
    pixels[:, 32:, 0] = 150
    out = io.BytesIO()
    Image.fromarray(pixels).save(out, format="JPEG")
    return out.getvalue()


def test_first_upload_cache_miss_runs_inference(client, api, image_bytes):
    """Test A: First upload -> cache MISS -> model inference occurs."""
    assert api.model.predict_calls == 0

    response = client.post(
        "/predict",
        files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "Forest"
    assert data["status"] == "KNOWN"
    # Model inference MUST have run exactly once
    assert api.model.predict_calls == 1


def test_second_identical_upload_cache_hit_skips_inference(client, api, image_bytes):
    """Test B: Same exact image -> cache HIT -> model inference does NOT occur."""
    # First upload (Miss)
    resp1 = client.post(
        "/predict",
        files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert resp1.status_code == 200
    assert api.model.predict_calls == 1

    # Second upload with identical bytes (Hit)
    resp2 = client.post(
        "/predict",
        files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert resp2.status_code == 200
    # Model inference call count MUST NOT increment
    assert api.model.predict_calls == 1

    # Responses must match
    assert resp1.json() == resp2.json()


def test_same_image_different_filename_is_cache_hit(client, api, image_bytes):
    """Test C: Same image with different filename -> cache HIT."""
    # Prime cache with original filename
    client.post(
        "/predict",
        files={"file": ("original_name.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert api.model.predict_calls == 1

    # Upload with completely different filename
    response = client.post(
        "/predict",
        files={"file": ("renamed_satellite_copy.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert response.status_code == 200
    data = response.json()
    # Cache HIT: model must NOT be called again
    assert api.model.predict_calls == 1
    # Response filename must reflect the new upload filename
    assert data["filename"] == "renamed_satellite_copy.jpg"
    assert data["predicted_class"] == "Forest"


def test_modified_image_is_cache_miss_runs_inference(
    client, api, image_bytes, modified_image_bytes
):
    """Test D: Slightly modified image -> cache MISS -> model inference occurs."""
    # First image
    client.post(
        "/predict",
        files={"file": ("image_a.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert api.model.predict_calls == 1

    # Modified image has different SHA-256 hash
    assert compute_image_hash(image_bytes) != compute_image_hash(modified_image_bytes)

    # Upload modified image
    response = client.post(
        "/predict",
        files={"file": ("image_a_modified.jpg", modified_image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert response.status_code == 200
    # Cache MISS: model inference MUST run again
    assert api.model.predict_calls == 2


def test_model_version_change_causes_cache_miss(client, api, image_bytes):
    """Test E: Different model version -> cache MISS."""
    original_version = api.MODEL_VERSION

    try:
        # Prime cache with version 1
        api.MODEL_VERSION = "model_version_v1"
        client.post(
            "/predict",
            files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
            params={"threshold": 0.3},
        )
        assert api.model.predict_calls == 1

        # Simulate model upgrade/change
        api.MODEL_VERSION = "model_version_v2"

        # Upload exact same image again
        response = client.post(
            "/predict",
            files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
            params={"threshold": 0.3},
        )
        assert response.status_code == 200
        # Cache MISS: different model version must invalidate old cache
        assert api.model.predict_calls == 2

    finally:
        api.MODEL_VERSION = original_version


def test_bulk_zip_with_duplicates_skips_repeated_inference(
    client, api, image_bytes, secondary_image_bytes
):
    """Test F: Bulk ZIP containing duplicate images -> deduplicates inference."""
    # Create a zip containing 3 copies of image A and 1 copy of image B (4 total)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("img_a_1.jpg", image_bytes)
        zf.writestr("img_a_2.jpg", image_bytes)
        zf.writestr("img_a_3.jpg", image_bytes)
        zf.writestr("img_b.jpg", secondary_image_bytes)

    response = client.post(
        "/predict/bulk",
        files={"file": ("batch.zip", buf.getvalue(), "application/zip")},
        params={"threshold": 0.3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4
    assert len(data["results"]) == 4

    # Inference MUST run only once for the batch of 2 unique images
    assert api.model.predict_calls == 1

    # Now verify that subsequent single upload of image_bytes is a cache HIT
    single_resp = client.post(
        "/predict",
        files={"file": ("check_cached.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.3},
    )
    assert single_resp.status_code == 200
    # No additional model calls!
    assert api.model.predict_calls == 1


def test_dynamic_threshold_on_cached_prediction(client, api, image_bytes):
    """Test G: Changing threshold on a cached image updates the status without inference."""
    # First upload with threshold 0.1 -> KNOWN
    resp1 = client.post(
        "/predict",
        files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.1},
    )
    assert resp1.status_code == 200
    assert resp1.json()["status"] == "KNOWN"
    assert api.model.predict_calls == 1

    # Second upload with extreme threshold 0.01 (P(Unknown)=0.05 >= 0.01 -> REVIEW)
    resp2 = client.post(
        "/predict",
        files={"file": ("scene.jpg", image_bytes, "image/jpeg")},
        params={"threshold": 0.01},
    )
    assert resp2.status_code == 200
    # Status dynamically updated based on new threshold
    assert resp2.json()["status"] == "REVIEW"
    assert resp2.json()["final_result"] == "UNRECOGNIZED"
    # But model inference was NOT executed!
    assert api.model.predict_calls == 1


def test_blank_image_is_not_cached(client, api):
    """Test H: Blank images return REVIEW and are not placed into cache."""
    output = io.BytesIO()
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(output, format="PNG")
    blank_bytes = output.getvalue()

    resp = client.post(
        "/predict",
        files={"file": ("blank.png", blank_bytes, "image/png")},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "REVIEW"

    # Verify not in cache
    img_hash = compute_image_hash(blank_bytes)
    assert api.prediction_cache.get(img_hash, api.MODEL_VERSION) is None


def test_sqlite_cache_direct(tmp_path):
    """Test I1: Direct unit tests for SQLitePredictionCache."""
    db_file = tmp_path / "test_cache.db"
    cache = SQLitePredictionCache(db_path=db_file)

    img_hash = "abc123def456"
    version = "v1"
    probs = [0.8, 0.1, 0.05, 0.03, 0.02]

    # Miss
    assert cache.get(img_hash, version) is None

    # Set
    success = cache.set(img_hash, version, probs, "Forest")
    assert success is True

    # Hit
    retrieved = cache.get(img_hash, version)
    assert retrieved is not None
    assert np.allclose(retrieved, probs)

    # Different version miss
    assert cache.get(img_hash, "v2") is None

    # Clear
    cache.clear()
    assert cache.get(img_hash, version) is None


def test_in_memory_cache_direct():
    """Test I2: Direct unit tests for InMemoryPredictionCache."""
    cache = InMemoryPredictionCache()
    img_hash = "hash_123"
    version = "ver_1"
    probs = [0.5, 0.2, 0.1, 0.1, 0.1]

    assert cache.get(img_hash, version) is None
    cache.set(img_hash, version, probs, "SeaLake")
    assert cache.get(img_hash, version) == probs
    cache.clear()
    assert cache.get(img_hash, version) is None


def test_azure_table_cache_unit():
    """Test I3: Direct unit tests for AzureTablePredictionCache with mocked TableClient."""
    mock_client = MagicMock()
    cache = AzureTablePredictionCache(table_name="testcache")
    cache._table_client = mock_client

    img_hash = "mock_hash_456"
    version = "v1.0"
    probs = [0.70, 0.10, 0.08, 0.07, 0.05]

    # Test set
    cache.set(img_hash, version, probs, "Forest")
    assert mock_client.upsert_entity.called
    call_args = mock_client.upsert_entity.call_args[1]
    entity = call_args["entity"]
    assert entity["PartitionKey"] == version
    assert entity["RowKey"] == img_hash
    assert json.loads(entity["probabilities"]) == probs
    assert entity["predicted_class"] == "Forest"

    # Test get (hit)
    mock_client.get_entity.return_value = {
        "probabilities": json.dumps(probs),
        "predicted_class": "Forest",
    }
    result = cache.get(img_hash, version)
    assert result == probs
    mock_client.get_entity.assert_called_with(
        partition_key=version,
        row_key=img_hash,
    )
