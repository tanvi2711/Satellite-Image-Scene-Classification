# 🛰️ Satellite Image Scene Classification

> **Production-oriented deep learning application for satellite/aerial scene classification using TensorFlow/Keras, FastAPI, Streamlit, Docker, GitHub Actions, and Microsoft Azure.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Azure](https://img.shields.io/badge/Microsoft%20Azure-Cloud-0078D4?logo=microsoftazure&logoColor=white)](https://azure.microsoft.com/)

---

## 📌 Overview

This project is an end-to-end **satellite/aerial image scene classification system**.

It accepts individual images or ZIP archives containing multiple images, preprocesses them, performs TensorFlow/Keras inference, evaluates prediction confidence, and returns the predicted scene class with a review status when confidence is low.

The application is packaged with Docker and deployed on **Microsoft Azure Container Apps**, with GitHub Actions providing automated CI/CD.

### Supported scene classes

| Class |
|---|
| 🌲 Forest |
| 🌊 SeaLake |
| 🏜️ Desert |
| ☁️ Cloudy |
| ❓ Unknown |

---

## ✨ Key Features

### 🖼️ Single Image Prediction
- Upload a satellite/aerial image.
- Automatic preprocessing and normalization.
- TensorFlow/Keras inference.
- Predicted class and confidence score.
- Low-confidence review handling.

### 📦 Bulk ZIP Prediction
- Upload a ZIP containing multiple images.
- Validate and process supported image files.
- Batched inference for improved efficiency.
- Per-image prediction and confidence.
- Accepted / review status.
- Results table.
- CSV export.

### 🎯 Confidence & Review Handling
Predictions are evaluated against a configurable confidence threshold.

A prediction that does not meet the configured criteria can be marked:

**Needs Review**

This provides a simple human-review mechanism instead of treating every prediction as equally reliable.

### 🔄 Test-Time Augmentation

The backend uses four inference views:

1. Original image
2. Horizontal flip
3. Vertical flip
4. 180° rotation

The resulting probabilities are averaged to produce the final prediction.

### ⚡ Batched Inference

Bulk inference uses a configured batch size of **32**.

```text
ZIP
 ↓
Read images
 ↓
Preprocess
 ↓
Generate 4 TTA views
 ↓
Batch TensorFlow inference
 ↓
Average TTA probabilities
 ↓
Generate per-image results
```

### 📊 Structured Logging

Backend logs capture prediction metadata such as:

- Request ID
- Prediction mode
- Filename
- Predicted class
- Confidence
- Unknown probability
- Status
- Review result
- Threshold
- Processing time

Raw image contents are not written to prediction logs.

---

# 🧠 Machine Learning

## Model

The classification system uses:

- TensorFlow
- Keras
- CNN-based image classification
- Transfer learning
- Regularization
- Training-time augmentation
- Test-Time Augmentation during inference

### Model files

```text
model/
├── satellite_v5_final.keras
└── config.json
```

### Input size

```text
224 × 224 pixels
```

---

# 📊 Model Performance

The current evaluation uses **400 labeled known-class test images**, with 100 images per known class.

## ⭐ Known-Class Accuracy

# **95.25%**

**381 / 400** predictions were correct.

**19 / 400** predictions were incorrect.

### Classification metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Forest | 100.00% | 95.00% | 97.44% | 100 |
| SeaLake | 96.00% | 96.00% | 96.00% | 100 |
| Desert | 96.00% | 96.00% | 96.00% | 100 |
| Cloudy | 100.00% | 94.00% | 96.91% | 100 |
| Unknown | 0.00% | 0.00% | 0.00% | 0 |

### Overall metrics

| Metric | Score |
|---|---:|
| Accuracy | **95.25%** |
| Weighted Precision | **98.00%** |
| Weighted Recall | **95.25%** |
| Weighted F1-Score | **96.59%** |

> **Important:** 95.25% is the accuracy on the 400 labeled **known-class** test images. The reported evaluation set contains no labeled `Unknown` samples, so this should not be described as overall five-class accuracy.

### Per-class performance

- **Forest:** F1 = 97.44%
- **SeaLake:** F1 = 96.00%
- **Desert:** F1 = 96.00%
- **Cloudy:** F1 = 96.91%

---

# 🏗️ System Architecture

```text
                         ┌─────────────────────┐
                         │        USER         │
                         │  Desktop / Mobile   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Streamlit Frontend  │
                         │                     │
                         │ • Single Upload     │
                         │ • Bulk ZIP Upload   │
                         │ • Results / CSV     │
                         └──────────┬──────────┘
                                    │ HTTP
                                    ▼
                         ┌─────────────────────┐
                         │   FastAPI Backend   │
                         │                     │
                         │ /health             │
                         │ /predict            │
                         │ /predict/bulk       │
                         │ /config             │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Image Preprocessing │
                         │                     │
                         │ 224 × 224           │
                         │ Normalization       │
                         │ TTA                 │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ TensorFlow / Keras  │
                         │ Classification      │
                         │ Model               │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Prediction Result   │
                         │                     │
                         │ Class               │
                         │ Confidence          │
                         │ Review Status       │
                         └─────────────────────┘
```

---

# ☁️ Azure Deployment Architecture

```text
                    ┌─────────────────────┐
                    │       GitHub        │
                    │     Repository      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   GitHub Actions    │
                    │                     │
                    │ Ruff / Mypy         │
                    │ Pytest / Coverage   │
                    │ Docker Build        │
                    └──────────┬──────────┘
                               │
                         Merge to main
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Azure Container     │
                    │ Registry (ACR)      │
                    │                     │
                    │ Backend Image       │
                    │ Frontend Image      │
                    └──────────┬──────────┘
                               │
                               ▼
             ┌──────────────────────────────────┐
             │      Azure Container Apps        │
             │                                  │
             │  ┌────────────────────────────┐  │
             │  │ satellite-frontend         │  │
             │  │ Streamlit                  │  │
             │  └─────────────┬──────────────┘  │
             │                │                 │
             │                ▼                 │
             │  ┌────────────────────────────┐  │
             │  │ satellite-backend          │  │
             │  │ FastAPI + TensorFlow       │  │
             │  └────────────────────────────┘  │
             └────────────────┬─────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Log Analytics     │
                    │                     │
                    │ Application Logs    │
                    │ Prediction Logs     │
                    │ Error Logs          │
                    └─────────────────────┘
```

## Azure services

| Service | Purpose |
|---|---|
| Azure Container Apps | Hosts frontend and backend |
| Azure Container Registry | Stores Docker images |
| Azure Blob Storage | Provisioned project storage |
| Azure Key Vault | Secure secret-management infrastructure |
| Managed Identity | Azure resource authentication |
| Azure RBAC | Access control |
| Log Analytics | Application/container logging |

### Production application flow

```text
GitHub
  ↓
GitHub Actions
  ↓
Azure OIDC Authentication
  ↓
Build Docker Images
  ↓
Push Images to ACR
  ↓
Update Container Apps
  ↓
Production
```

---

# 🔀 Git Branching Strategy

The project follows a feature-based Git workflow:

```text
main
 │
 ▼
develop
 │
 ├── feature/*
 ├── bugfix/*
 └── hotfix/*
```

Typical development flow:

```text
Feature Branch
      ↓
Pull Request
      ↓
Automated CI
      ↓
develop
      ↓
Pull Request
      ↓
main
      ↓
Automated CD
      ↓
Azure
```

Conventional commit categories used include:

```text
feat
fix
docs
test
refactor
style
chore
```

---

# 🔄 CI/CD

## Continuous Integration

GitHub Actions validates changes using:

- Ruff linting
- Ruff formatting checks
- Mypy type checking
- Pytest
- Coverage validation
- Docker image builds

CI runs for relevant pushes and Pull Requests.

## Continuous Deployment

After changes reach `main`:

```text
Merge to main
      ↓
GitHub Actions
      ↓
Azure OIDC Login
      ↓
ACR Login
      ↓
Build Backend Image
      ↓
Build Frontend Image
      ↓
Push Images
      ↓
Update Backend Container App
      ↓
Update Frontend Container App
      ↓
Production Deployment
```

---

# 🐳 Docker

The project uses separate containers for the frontend and backend.

## Backend

Includes:

- Python
- FastAPI
- TensorFlow CPU
- Keras
- Pillow
- NumPy
- Pydantic
- Backend application
- Trained model

**Port:** `8000`

## Frontend

Includes:

- Python
- Streamlit
- Requests
- Pandas
- Frontend application

**Port:** `8501`

Run locally with Docker Compose:

```bash
docker compose up --build
```

---

# 📁 Project Structure

```text
Satellite-Image-Scene-Classification/
│
├── backend/
│   ├── main.py
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── model/
│   ├── satellite_v5_final.keras
│   └── config.json
│
├── evaluation/
│   └── evaluate_model.py
│
├── tests/
│   ├── conftest.py
│   ├── test_health.py
│   ├── test_predict.py
│   └── test_bulk.py
│
├── .github/
│   └── workflows/
│
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── README.md
└── ...
```

---

# 🚀 Local Setup

## Prerequisites

Install:

- Python 3.11
- Git
- Docker Desktop
- Docker Compose

## 1. Clone

```bash
git clone https://github.com/tanvi2711/Satellite-Image-Scene-Classification.git
cd Satellite-Image-Scene-Classification
```

## 2. Backend

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Start FastAPI:

```bash
uvicorn backend.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

## 3. Frontend

Install dependencies:

```bash
pip install -r frontend/requirements.txt
```

Set the backend URL in PowerShell:

```powershell
$env:BACKEND_URL="http://localhost:8000"
```

Start Streamlit:

```bash
streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

---

# 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Backend health check |
| `POST` | `/predict` | Single-image prediction |
| `POST` | `/predict/bulk` | ZIP/bulk prediction |
| `GET` | `/config` | Prediction configuration |

### `POST /predict`

Accepts an uploaded image and returns:

- Predicted class
- Confidence
- Status/review information

### `POST /predict/bulk`

Accepts a ZIP archive containing images and returns per-image results.

---

# 🧪 Testing

Automated tests cover:

- Health endpoint
- Single-image prediction
- Bulk prediction

```text
tests/
├── conftest.py
├── test_health.py
├── test_predict.py
└── test_bulk.py
```

Run tests:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov
```

---

# 📈 Evaluation Workflow

```text
Test Dataset
     ↓
Image Preprocessing
     ↓
Model Prediction
     ↓
Predicted Probabilities
     ↓
Class Selection
     ↓
Confidence Evaluation
     ↓
Classification Metrics
     ↓
Confusion Matrix
```

Reported metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

---

# ⚡ Performance Optimization

Bulk inference was optimized from repeated individual model prediction calls to batched inference.

Current pipeline:

```text
ZIP
 ↓
Read Images
 ↓
Preprocess
 ↓
Generate 4 TTA Views
 ↓
Batch TensorFlow Prediction
 ↓
Average Probabilities
 ↓
Generate Individual Results
```

Configured batch size:

```text
32
```

This reduces repeated model invocation overhead while keeping batch memory bounded.

---

# 📱 Production Testing

The deployed application has been tested with:

- Desktop browser
- Mobile browser
- Single image upload
- JPG image upload
- Bulk ZIP upload
- Prediction results
- Confidence display
- Low-confidence handling

A Streamlit session-routing issue affecting production file uploads was resolved by enabling **session affinity** on the frontend Azure Container App.

The production upload flow was retested successfully after the configuration change.

---

# 📊 Monitoring & Logging

Backend logs are written to standard output and collected by Azure Log Analytics.

Example fields include:

```text
event
request_id
mode
filename
predicted_class
confidence
confidence_percent
p_unknown
status
final_result
threshold
processing_time_ms
```

Bulk processing also records batch-level information.

This provides operational visibility while avoiding raw image content in prediction logs.

---

# 🔐 Security

Security practices include:

- No API keys or credentials committed to Git.
- GitHub Actions uses Azure OIDC authentication.
- Managed Identity is used for Azure resource authentication.
- Azure RBAC controls resource access.
- Azure Key Vault is provisioned for secure secret management.
- Prediction logs contain metadata rather than raw image data.

---

# 📋 BRD Alignment

The project addresses the core application and engineering requirements defined in the BRD.

### Functional requirements

| Requirement | Status |
|---|---|
| Multi-class classification | ✅ |
| Image preprocessing | ✅ |
| Model evaluation | ✅ |
| Unseen-image prediction | ✅ |
| Single-image upload | ✅ |
| Bulk ZIP upload | ✅ |
| Confidence score | ✅ |
| Bulk result table | ✅ |
| Configurable threshold | ✅ |
| Low-confidence review | ✅ |
| Unknown/unclassified handling | ✅ |

### Engineering requirements

| Requirement | Status |
|---|---|
| Automated linting | ✅ |
| Type checking | ✅ |
| Unit testing | ✅ |
| Coverage validation | ✅ |
| No hardcoded secrets | ✅ |
| Docker | ✅ |
| Git branching | ✅ |
| Pull Requests | ✅ |
| CI | ✅ |
| CD | ✅ |
| Azure deployment | ✅ |
| Managed Identity / RBAC | ✅ |
| Azure logging | ✅ |

### Current BRD considerations

**Azure Blob Storage:** The Azure Blob Storage environment has been provisioned. The current prediction flow directly processes user uploads through Streamlit and FastAPI rather than using Blob Storage as an intermediate upload store.

**Grad-CAM:** Grad-CAM/explainability was identified as an optional requirement and is not part of the current production implementation.

These items are therefore not represented as completed features.

---

# 🔮 Future Improvements

Potential future enhancements:

1. Dedicated Azure Blob Storage ingestion pipeline.
2. Dedicated labeled `Unknown` / OOD evaluation dataset.
3. Confidence calibration.
4. Grad-CAM explainability.
5. Additional monitoring dashboards.
6. Automated model versioning.
7. Experiment tracking.
8. Further inference optimization.
9. Expanded scene categories.
10. Larger and more diverse evaluation datasets.

---

# 🎯 Project Outcome

This project demonstrates an end-to-end machine learning engineering workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Model Training
   ↓
Model Evaluation
   ↓
FastAPI Inference API
   ↓
Streamlit Web Application
   ↓
Docker
   ↓
Automated Testing
   ↓
GitHub Actions CI/CD
   ↓
Azure Container Registry
   ↓
Azure Container Apps
   ↓
Production Application
   ↓
Monitoring / Logging
```

The result is a production-oriented satellite scene classification application combining:

**Machine Learning + Backend Development + Web UI + Docker + CI/CD + Cloud Deployment + Monitoring**

---

# 👤 Author

**Tanvi Jivatode**

**Project:** Satellite Image Scene Classification

---

## 📌 Project Status

**Production Internship Submission**

- ✅ Trained classification model
- ✅ Model evaluation
- ✅ 95.25% known-class test accuracy
- ✅ Single-image prediction
- ✅ Bulk ZIP prediction
- ✅ Confidence scoring
- ✅ Configurable review threshold
- ✅ Unknown / unclassified handling
- ✅ CSV export
- ✅ Streamlit UI
- ✅ FastAPI backend
- ✅ Batched bulk inference
- ✅ Structured logging
- ✅ Automated tests
- ✅ Linting
- ✅ Type checking
- ✅ Docker
- ✅ Git branching and Pull Requests
- ✅ GitHub Actions CI
- ✅ Automated Azure CD
- ✅ Azure Container Registry
- ✅ Azure Container Apps
- ✅ Azure Managed Identity
- ✅ Azure RBAC
- ✅ Azure Key Vault provisioned
- ✅ Azure Log Analytics
- ✅ Production deployment
- ✅ Mobile upload tested
- ✅ Production upload issue resolved

---

## 📄 License

This project was developed as part of an internship project and is intended for educational, demonstration, and internship evaluation purposes.
