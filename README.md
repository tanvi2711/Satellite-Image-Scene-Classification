# 🛰️ Satellite Image Scene Classification

A deep-learning web application that classifies satellite and aerial images into **Forest, SeaLake, Desert, Cloudy, or Unknown**.

Built with **TensorFlow/Keras, FastAPI, Streamlit, Docker, GitHub Actions, and Microsoft Azure**.

---

## 👩‍💻 Author

**Tanvi Jivatode**

---

## 🎯 Project Overview

The system provides an end-to-end workflow for satellite scene classification:

```text
Image / ZIP Upload
        ↓
Preprocessing
        ↓
EfficientNetV2-S Model
        ↓
Prediction Probabilities
        ↓
Confidence / Unknown Check
        ↓
Final Class + Review Status
```

### Main features

- 🖼️ Single-image prediction
- 📦 Bulk ZIP prediction
- 🎯 Confidence score
- ⚠️ Low-confidence review status
- ❓ Unknown / unclassified handling
- 🔄 Test-Time Augmentation (TTA)
- ⚡ Batched bulk inference
- 📊 Results table and CSV export
- 🧪 Automated tests
- 🐳 Docker deployment
- 🔄 GitHub Actions CI/CD
- ☁️ Microsoft Azure deployment
- 📈 Structured application logging

---

# 🧠 Machine Learning Model

The project uses **EfficientNetV2-S** with **ImageNet pretrained weights**.

### Model architecture

```text
Input Image
     ↓
224 × 224 × 3
     ↓
Image Augmentation
     ↓
EfficientNetV2-S Backbone
     ↓
Global Average Pooling
     ↓
Dropout
     ↓
Dense Layer
     ↓
Softmax
     ↓
5 Classes
```

### Model techniques

- Transfer learning
- ImageNet pretrained weights
- Dropout regularization
- L2 regularization
- Label smoothing
- Class weighting
- Training-time augmentation
- Two-stage training
- Test-Time Augmentation

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

# 📚 Dataset

The main dataset used is **NWPU-RESISC45**.

Four NWPU scene categories are used as the known classes:

| NWPU Category | Project Class |
|---|---|
| Forest | Forest |
| Lake | SeaLake |
| Desert | Desert |
| Cloudy | Cloudy |

Other visually different NWPU satellite scene categories are used as **Unknown / hard-negative examples**.

Examples include scenes such as:

- River
- Mountain
- Beach
- Meadow
- Wetland
- Island
- Snowberg
- Sea ice
- Farmland

The purpose is to teach the model that not every satellite image belongs to one of the four target classes.

---

# ❓ Unknown Class

`Unknown` is handled as a rejection category rather than simply another normal scene.

Unknown training examples can include:

- Other NWPU scene classes
- Synthetic negative images
- Optional CIFAR-100 images

Synthetic negatives include examples such as:

- Blank images
- Random noise
- Gradients
- Shapes
- Text/document-like images
- Stripes
- Checker patterns

### Unknown decision

The prediction logic considers the probability of the Unknown class.

Conceptually:

```text
Top prediction = known class
        AND
P(Unknown) < configured threshold
        ↓
       ACCEPT
```

Otherwise:

```text
       REVIEW
```

The rejection threshold is selected using a separate calibration process rather than directly using the final test set.

---

# 🔄 Test-Time Augmentation

The backend uses four prediction views:

```text
1. Original
2. Horizontal Flip
3. Vertical Flip
4. 180° Rotation
```

The model predictions from these views are averaged.

```text
Input Image
     ↓
4 TTA Views
     ↓
Model Predictions
     ↓
Average Probabilities
     ↓
Final Prediction
```

This provides additional robustness during inference.

---

# 🏋️ Training

Training is performed in two main stages.

### Phase 1 — Classification Head

- EfficientNetV2-S backbone is initially frozen.
- Classification head is trained.
- AdamW optimization.
- Label smoothing.
- Class weighting.
- Early stopping.
- Learning-rate reduction.

### Phase 2 — Fine-Tuning

- Best Phase 1 model is loaded.
- Selected EfficientNetV2-S layers are fine-tuned.
- Batch-normalization layers remain protected.
- Cosine learning-rate scheduling is used.
- Early stopping is applied.

This allows the pretrained network to first learn the new scene-classification task and then adapt its features to satellite imagery.

---

# 🖼️ Training Augmentation

The training pipeline applies multiple transformations to improve robustness, including:

- Random crop
- Aspect-ratio / zoom variation
- Horizontal and vertical flips
- Rotation
- Translation
- Brightness changes
- Contrast changes
- Saturation changes
- Hue changes
- Colour-cast simulation
- JPEG quality changes
- Low-resolution simulation
- Blur
- Noise
- Overlay / watermark-like effects

---

# 📊 Model Evaluation

The reported known-class evaluation uses:

```text
400 labeled test images
100 images per known class
```

## ⭐ Known-Class Accuracy

# **95.25%**

```text
381 / 400 correct
19 / 400 incorrect
```

### Classification results

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Forest | 100.00% | 95.00% | 97.44% |
| SeaLake | 96.00% | 96.00% | 96.00% |
| Desert | 96.00% | 96.00% | 96.00% |
| Cloudy | 100.00% | 94.00% | 96.91% |

### Overall metrics

| Metric | Score |
|---|---:|
| Accuracy | **95.25%** |
| Weighted Precision | **98.00%** |
| Weighted Recall | **95.25%** |
| Weighted F1 | **96.59%** |

> **Important:** 95.25% is the accuracy on the 400 labeled **known-class** test images. It should not be described as overall five-class accuracy because the reported test set does not contain labeled `Unknown` samples.

---

# 🌐 Application Architecture

```text
                 USER
                  │
                  ▼
        ┌──────────────────┐
        │ Streamlit        │
        │ Frontend         │
        │                  │
        │ Single / Bulk    │
        └────────┬─────────┘
                 │ HTTP
                 ▼
        ┌──────────────────┐
        │ FastAPI Backend  │
        │                  │
        │ /predict         │
        │ /predict/bulk    │
        │ /health          │
        │ /config          │
        └────────┬─────────┘
                 │
                 ▼
        ┌──────────────────┐
        │ TensorFlow/Keras │
        │ EfficientNetV2-S │
        └────────┬─────────┘
                 │
                 ▼
        Prediction + Confidence
                 │
                 ▼
           ACCEPT / REVIEW
```

---

# 📦 Bulk Prediction

Users can upload a ZIP file containing multiple images.

```text
ZIP Upload
    ↓
Extract Images
    ↓
Validate Images
    ↓
Preprocess
    ↓
Create TTA Views
    ↓
Batch Inference
    ↓
Prediction Results
    ↓
Results Table
    ↓
CSV Download
```

Bulk inference uses a batch size of **32** to reduce repeated model-inference overhead.

---

# ☁️ Azure Deployment

The application is containerized with Docker and deployed on **Microsoft Azure Container Apps**.

### Main technologies/services

| Technology | Purpose |
|---|---|
| Streamlit | Web frontend |
| FastAPI | Backend API |
| TensorFlow/Keras | Model inference |
| Docker | Containerization |
| Azure Container Registry | Docker image storage |
| Azure Container Apps | Production hosting |
| Azure Log Analytics | Application/container logs |
| Managed Identity | Secure Azure authentication |
| Azure RBAC | Access control |
| GitHub Actions | CI/CD |

### Deployment flow

```text
GitHub
   ↓
GitHub Actions
   ↓
Lint + Type Check + Tests
   ↓
Docker Build
   ↓
Azure Container Registry
   ↓
Azure Container Apps
   ↓
Production
```

---

# 🔐 Security

The project follows basic production security practices:

- No API keys or credentials committed to Git.
- GitHub Actions uses Azure OIDC authentication.
- Azure Managed Identity is used for Azure access.
- Azure RBAC controls permissions.
- Azure Key Vault is provisioned for secret management.
- Prediction logs contain metadata rather than raw image data.

---

# 📈 Logging & Monitoring

The backend produces structured logs containing information such as:

```text
request_id
mode
filename
predicted_class
confidence
p_unknown
status
threshold
processing_time_ms
```

Azure Log Analytics is used to inspect application and container logs.

---

# 🔀 Git & CI/CD

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

Typical workflow:

```text
Feature Branch
      ↓
Pull Request
      ↓
CI Checks
      ↓
develop
      ↓
Pull Request
      ↓
main
      ↓
CD
      ↓
Azure
```

### CI checks

- Ruff linting
- Ruff formatting
- Mypy type checking
- Pytest
- Coverage validation
- Docker builds

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
└── README.md
```

---

# 🚀 Local Setup

## Requirements

- Python 3.11
- Git
- Docker Desktop
- Docker Compose

### Clone

```bash
git clone https://github.com/tanvi2711/Satellite-Image-Scene-Classification.git
cd Satellite-Image-Scene-Classification
```

### Backend

```bash
python -m venv venv
venv\Scripts\activate

pip install -r backend/requirements.txt

uvicorn backend.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

### Frontend

```powershell
$env:BACKEND_URL="http://localhost:8000"
streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

### Docker

```bash
docker compose up --build
```

---

# 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Backend health check |
| POST | `/predict` | Predict one image |
| POST | `/predict/bulk` | Predict images from ZIP |
| GET | `/config` | View prediction configuration |

---

# ⚠️ Current Limitations

- The reported **95.25%** metric is known-class accuracy.
- The reported test set contains no labeled `Unknown` samples.
- A larger dedicated Unknown/OOD benchmark would improve rejection evaluation.
- Grad-CAM explainability is not currently implemented.
- Azure Blob Storage is provisioned, but the current prediction flow sends uploaded images directly through **Streamlit → FastAPI** rather than using Blob Storage as an intermediate upload store.

---

# 🔮 Future Improvements

- Dedicated Unknown/OOD evaluation dataset.
- Confidence calibration improvements.
- Grad-CAM explanations.
- Azure Blob-based ingestion pipeline.
- Model versioning and experiment tracking.
- More satellite scene categories.
- Larger and more diverse datasets.
- Further inference optimization.

---

# ✅ Project Status

**Production Internship Project**

- ✅ EfficientNetV2-S classification model
- ✅ NWPU-RESISC45-based training
- ✅ Unknown / rejection handling
- ✅ 95.25% known-class test accuracy
- ✅ Single-image prediction
- ✅ Bulk ZIP prediction
- ✅ Confidence scoring
- ✅ Test-Time Augmentation
- ✅ Batched inference
- ✅ FastAPI backend
- ✅ Streamlit frontend
- ✅ Docker
- ✅ Automated testing
- ✅ GitHub Actions CI/CD
- ✅ Azure deployment
- ✅ Azure logging and monitoring

---

## 👩‍💻 Author

**Tanvi Jivatode**

**Satellite Image Scene Classification**
