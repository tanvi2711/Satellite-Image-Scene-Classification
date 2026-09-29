def test_health_returns_model_status(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True

    assert data["classes"] == [
        "Forest",
        "SeaLake",
        "Desert",
        "Cloudy",
        "Unknown"
    ]

    assert data["threshold"] == 0.3


def test_config_returns_known_classes(client):
    response = client.get("/config")

    assert response.status_code == 200

    data = response.json()

    assert data["class_names"] == [
        "Forest",
        "SeaLake",
        "Desert",
        "Cloudy"
    ]

    assert data["default_threshold"] == 0.3