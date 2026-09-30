import io
import zipfile


def make_zip(entries):
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)

    return buffer.getvalue()


def test_bulk_predict_processes_images(client, image_bytes):
    archive = make_zip([("images/one.jpg", image_bytes), ("two.jpg", image_bytes)])

    response = client.post(
        "/predict/bulk",
        files={"file": ("images.zip", archive, "application/zip")},
        params={"threshold": 0.3},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 2
    assert data["known"] == 2
    assert data["review"] == 0

    assert [item["filename"] for item in data["results"]] == ["one.jpg", "two.jpg"]

    assert all(item["status"] == "KNOWN" for item in data["results"])


def test_bulk_rejects_non_zip_file(client):
    response = client.post(
        "/predict/bulk",
        files={"file": ("images.txt", b"not a zip", "application/octet-stream")},
    )

    assert response.status_code == 400

    assert response.json()["detail"] == ("File must be a .zip archive.")


def test_bulk_rejects_corrupt_zip(client):
    response = client.post(
        "/predict/bulk", files={"file": ("images.zip", b"not a zip", "application/zip")}
    )

    assert response.status_code == 400

    assert response.json()["detail"] == ("Invalid zip file.")


def test_bulk_rejects_zip_without_images(client):
    archive = make_zip([("readme.txt", b"No images here")])

    response = client.post(
        "/predict/bulk", files={"file": ("empty.zip", archive, "application/zip")}
    )

    assert response.status_code == 400

    assert response.json()["detail"] == ("No images found inside the zip.")


def test_bulk_reports_bad_image_as_review(client):
    archive = make_zip([("broken.jpg", b"not an image")])

    response = client.post(
        "/predict/bulk",
        files={"file": ("broken.zip", archive, "application/zip")},
        params={"threshold": 0.3},
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total"] == 1
    assert data["known"] == 0
    assert data["review"] == 1

    assert data["results"][0]["status"] == "REVIEW"
    assert data["results"][0]["predicted_class"] == "ERROR"
