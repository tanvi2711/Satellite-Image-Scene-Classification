# 🛰️ Satellite Image Scene Classification

A production-oriented deep learning application for classifying satellite/aerial scene images using a TensorFlow/Keras image-classification model.

The system provides a complete workflow from image upload and preprocessing to model inference, confidence evaluation, low-confidence review handling, bulk ZIP classification, automated testing, Dockerization, CI/CD, and deployment on Microsoft Azure.

---

## 📌 Project Overview

Satellite and aerial imagery can contain large amounts of information about different land-use and land-cover scenes. Manually analyzing large collections of images is time-consuming and difficult to scale.

This project develops an automated satellite scene classification system that classifies images into predefined scene categories and provides confidence information for every prediction.

The application supports:

- Single-image classification
- Bulk ZIP image classification
- Per-image confidence scores
- Configurable confidence threshold
- Low-confidence / review status
- Unknown / unclassified handling
- Bulk results table
- CSV export
- Test-Time Augmentation (TTA)
- Batched bulk inference
- Structured backend logging
- Automated testing
- Docker deployment
- GitHub Actions CI/CD
- Microsoft Azure deployment

---

# 🎯 Project Objectives

The main objectives of the project are:

1. Develop a multi-class satellite scene classification model.
2. Normalize and preprocess satellite image inputs.
3. Evaluate model performance using standard classification metrics.
4. Provide prediction capability for new and unseen images.
5. Build a web application for single-image prediction.
6. Build a bulk ZIP prediction workflow.
7. Display confidence scores for every prediction.
8. Flag low-confidence predictions for review.
9. Implement proper Git branching and Pull Request workflows.
10. Implement automated CI checks.
11. Implement automated deployment to Microsoft Azure.
12. Apply Docker-based application deployment.
13. Provide application logging and monitoring.
14. Provide documentation for setup, deployment, architecture, and results.

---

# ✨ Key Features

## 1. Single Image Classification

Users can upload a single satellite/aerial image through the web application.

The system:

1. Accepts the uploaded image.
2. Preprocesses the image.
3. Runs the trained TensorFlow/Keras model.
4. Applies prediction logic and confidence evaluation.
5. Returns the predicted scene class.
6. Displays the confidence score.
7. Indicates whether the result requires review.

---

## 2. Bulk ZIP Classification

Users can upload a ZIP file containing multiple satellite images.

The system:

1. Reads the ZIP archive.
2. Identifies valid image files.
3. Extracts and preprocesses the images.
4. Creates Test-Time Augmentation views.
5. Performs batched inference.
6. Generates a result for every valid image.
7. Displays per-image prediction and confidence.
8. Flags low-confidence results.
9. Provides a results table.
10. Allows the results to be downloaded as CSV.

---

## 3. Confidence Score

Every prediction includes a confidence score.

Example:

```text
Forest
Confidence: 95.20%
The confidence information helps users understand how strongly the model supports a prediction.
4. Configurable Confidence Threshold
The application supports a configurable threshold for identifying predictions that require review.
Predictions that do not meet the configured confidence criteria can be marked as:
Needs Review

This provides a human-review mechanism instead of treating every prediction as equally reliable.
5. Unknown / Unclassified Handling
The model includes an Unknown category.
The prediction logic can also identify cases where an image should not be treated as a confident known-class prediction.
This helps reduce forced classification of images that may not sufficiently resemble the known scene categories.
6. Test-Time Augmentation
The backend uses four prediction views for Test-Time Augmentation:
1. Original image
2. Horizontal flip
3. Vertical flip
4. 180-degree rotation
The model predictions from these views are averaged to produce the final prediction probabilities.
This provides additional robustness during inference.
7. Batched Bulk Inference
The bulk prediction pipeline was optimized to avoid running an independent model inference operation for every image.
Instead, the system:
Images
   ↓
Preprocessing
   ↓
TTA Views
   ↓
Batch Construction
   ↓
TensorFlow Inference
   ↓
Average TTA Probabilities
   ↓
Per-image Results

The configured bulk inference batch size is:
32

This reduces repeated inference overhead while keeping memory usage bounded.
8. Structured Logging
The backend generates structured logs for prediction requests.
Logged information includes:
- Request ID
- Prediction mode
- Filename
- Predicted class
- Confidence
- Confidence percentage
- Unknown probability
- Status
- Final result
- Confidence threshold
- Processing time
Image contents are not written into prediction logs.
Bulk processing also records batch-level information.
🧠 Machine Learning Model
Model Framework
The classification model is implemented using:
- TensorFlow
- Keras
- Transfer learning
- CNN-based image classification
- Regularization
- Training-time augmentation
- Test-Time Augmentation during inference
Model Artifact
The trained model is stored at:
model/satellite_v5_final.keras

Model configuration:
model/config.json

Input Size
The model expects images normalized to:
224 × 224 pixels

🏷️ Scene Classes
The current model contains the following classes:
Forest
SeaLake
Desert
Cloudy
Unknown

📊 Model Evaluation
The current evaluation was performed using:
400 labeled known-class test images
100 images per known class

Overall Known-Class Accuracy
95.25%
381 / 400 correctly classified
19 / 400 incorrectly classified

Classification Results
Class	Precision	Recall	F1-Score	Support
Forest	100.00%	95.00%	97.44%	100
SeaLake	96.00%	96.00%	96.00%	100
Desert	96.00%	96.00%	96.00%	100
Cloudy	100.00%	94.00%	96.91%	100
Unknown	0.00%	0.00%	0.00%	0


Overall Metrics
Metric	Score
Accuracy	95.25%
Weighted Precision	98.00%
Weighted Recall	95.25%
Weighted F1-Score	96.59%


📌 Evaluation Interpretation
The model achieved:
95.25% accuracy on 400 known-class test images.
This corresponds to:
381 correct predictions
19 incorrect predictions

Per-class performance is consistently strong:
- Forest F1-score: 97.44%
- SeaLake F1-score: 96.00%
- Desert F1-score: 96.00%
- Cloudy F1-score: 96.91%
The Unknown class has zero support in this evaluation set because no labeled Unknown samples were included in the reported test dataset.
Therefore:
95.25% should be interpreted as known-class test accuracy on 400 labeled images, not as five-class accuracy on a test set containing labeled Unknown samples.

📈 Confusion Matrix
The confusion matrix shows strong diagonal performance across the evaluated known classes.
Known-class results:
Forest
95 / 100 correctly classified

SeaLake
96 / 100 correctly classified

Desert
96 / 100 correctly classified

Cloudy
94 / 100 correctly classified

Some known images are classified as Unknown, which is consistent with the application's conservative low-confidence/unclassified handling.
🏗️ System Architecture

                         ┌─────────────────────┐
                         │        User         │
                         │ Web / Mobile Browser │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Streamlit Frontend  │
                         │                     │
                         │ Single Upload       │
                         │ Bulk ZIP Upload     │
                         └──────────┬──────────┘
                                    │
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

☁️ Azure Architecture
The application is deployed using Microsoft Azure.
Main Azure services used/provisioned include:
- Azure Container Apps
- Azure Container Registry
- Azure Blob Storage
- Azure Key Vault
- Azure Managed Identity
- Azure Log Analytics
Production deployment flow:

                        GitHub Repository
                               │
                               ▼
                        GitHub Actions
                               │
                    ┌──────────┴──────────┐
                    │                     │
                 CI Checks            Docker Build
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                    Azure Container Registry
                               │
                    ┌──────────┴──────────┐
                    │                     │
             Backend Image        Frontend Image
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                    Azure Container Apps
                    ┌──────────┴──────────┐
                    │                     │
          satellite-backend       satellite-frontend
                    │                     │
                    └──────────┬──────────┘
                               │
                               ▼
                       Log Analytics
☁️ Azure Resources
Azure Container Registry
Container images are stored in Azure Container Registry.
Images include:
satellite-backend
satellite-frontend

Azure Container Apps
Two Container Apps are used:
satellite-backend
satellite-frontend

The frontend is externally accessible.
The backend is configured for internal application communication.
Azure Blob Storage
An Azure Blob Storage account and container have been provisioned for the project environment.
The BRD specifies Azure Blob Storage for satellite image data, processed data, and model artifacts.
The current end-user prediction flow, however, directly processes uploaded files through the web application and FastAPI backend rather than using Blob Storage as an intermediate upload store.
Azure Key Vault
Azure Key Vault has been provisioned for secure secret management.
The project avoids committing credentials or API keys to the Git repository.
Azure Managed Identity
Managed Identity is used for Azure resource authentication and supports the least-privilege access approach required by the BRD.
Azure Log Analytics
Container and application logs are collected and queried using Azure Log Analytics.
This provides visibility into:
- Application startup
- Health checks
- Predictions
- Bulk inference
- Processing times
- Errors
- Batch processing
🔐 Security
Security-related practices include:
- No API keys or credentials committed to Git.
- Azure federated identity/OIDC for GitHub Actions authentication.
- Managed Identity for Azure resource access.
- Azure RBAC for permissions.
- Azure Key Vault provisioned for secure secret management.
- Uploaded images are processed for prediction rather than intentionally stored permanently by the prediction application.
- Application logs record prediction metadata rather than raw image content.
🔀 Git Branching Strategy
The project follows a feature-based Git workflow.
main
 │
 ▼
develop
 │
 ├── feature/*
 ├── bugfix/*
 └── hotfix/*

Changes are developed on feature or bug-fix branches and merged using Pull Requests.
Typical workflow:
Feature Branch
      │
      ▼
Pull Request
      │
      ▼
Automated CI
      │
      ▼
develop
      │
      ▼
Pull Request
      │
      ▼
main
      │
      ▼
Automated CD
      │
      ▼
Azure

The project uses conventional commit categories such as:
feat
fix
docs
test
refactor
style
chore

🔄 CI/CD Pipeline
Continuous Integration
GitHub Actions performs automated quality checks.
The CI workflow includes:
- Ruff linting
- Ruff formatting checks
- Mypy type checking
- Pytest
- Coverage validation
- Docker image builds
CI checks are executed for relevant pushes and Pull Requests.
Continuous Deployment
The deployment workflow is triggered after changes reach the main branch.
The deployment process is:
Merge to main
      ↓
GitHub Actions
      ↓
Azure OIDC Authentication
      ↓
Azure Container Registry Login
      ↓
Build Backend Image
      ↓
Build Frontend Image
      ↓
Push Images to ACR
      ↓
Update Backend Container App
      ↓
Update Frontend Container App
      ↓
Production Deployment

The production CD workflow has been successfully executed.
🐳 Docker
The application is containerized for consistent deployment.
Backend Container
The backend container contains:
- Python
- FastAPI
- TensorFlow CPU
- Keras
- Pillow
- NumPy
- Pydantic
- Backend application
- Trained model
The backend exposes:
Port 8000

Frontend Container
The frontend container contains:
- Python
- Streamlit
- Requests
- Pandas
- Frontend application
The frontend exposes:
Port 8501

💻 Local Setup
Prerequisites
Install:
- Python 3.11
- Git
- Docker Desktop
- Docker Compose
1. Clone the Repository
git clone https://github.com/tanvi2711/Satellite-Image-Scene-Classification.git

Move into the project:
cd Satellite-Image-Scene-Classification

2. Backend Setup
Create a virtual environment:
python -m venv venv

Activate it on Windows:
venv\Scripts\activate

Install backend dependencies:
pip install -r backend/requirements.txt

Start the FastAPI server:
uvicorn backend.main:app --reload --port 8000

Backend URL:
http://localhost:8000

Health endpoint:
http://localhost:8000/health

3. Frontend Setup
Install frontend dependencies:
pip install -r frontend/requirements.txt

Set the backend URL in PowerShell:
$env:BACKEND_URL="http://localhost:8000"

Run Streamlit:
streamlit run frontend/app.py

Frontend URL:
http://localhost:8501

4. Run Using Docker Compose
Build and start both services:
docker compose up --build

The frontend communicates with the backend using the Docker Compose service network.
📁 Project Structure
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

🔌 API Endpoints
GET /health
Health-check endpoint used to verify that the backend service is running.
Example:
GET /health

POST /predict
Single-image prediction endpoint.
Example:
POST /predict

The endpoint accepts an uploaded image and returns prediction information including class, confidence and status.
POST /predict/bulk
Bulk ZIP prediction endpoint.
Example:
POST /predict/bulk

The endpoint accepts a ZIP file containing multiple images and returns per-image classification results.
GET /config
Returns prediction configuration information.
Example:
GET /config

🧪 Testing
Automated tests are included for:
- Backend health endpoint
- Single-image prediction
- Bulk prediction
Test files:
tests/
├── conftest.py
├── test_health.py
├── test_predict.py
└── test_bulk.py

Run tests:
pytest

Run tests with coverage:
pytest --cov

The CI pipeline also validates the project's automated test suite.

📊 Evaluation Workflow
The evaluation process includes:
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

Reported evaluation metrics include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
📋 Bulk Result Workflow
For every valid image inside a ZIP archive, the application produces information such as:
Filename
Predicted Class
Confidence
Status

Example:
image_001.jpg
Forest
95.20%
Accepted

or:
image_002.jpg
Unknown
18.40%
Needs Review

⚡ Performance Optimization
The original bulk inference approach performed repeated model prediction operations for individual images.
The implementation was optimized to use batched inference.
Current approach:
ZIP
 ↓
Read images
 ↓
Preprocess
 ↓
Generate 4 TTA views per image
 ↓
Combine views
 ↓
Batch TensorFlow prediction
 ↓
Average probabilities
 ↓
Generate individual results

The configured batch size is:
32

This reduces repeated model invocation overhead and improves bulk processing efficiency.

📱 Production Testing
The deployed application has been tested with:
- Desktop browser
- Mobile browser
- Single image upload
- JPG image upload
- Bulk ZIP upload
- Prediction results
- Confidence display
- Low-confidence handling
A Streamlit session-routing issue affecting production file uploads was resolved by enabling session affinity on the frontend Azure Container App.
The production upload flow has been retested successfully after the configuration change.

📈 Monitoring
The backend writes structured logs to standard output.
Azure Container Apps collects these logs through Log Analytics.
Example logged fields:
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

For bulk processing, batch-level information is also recorded.
This enables operational troubleshooting without storing raw image data in the logs.

📝 BRD Alignment
The project was developed against the requirements defined in the Business Requirements Document.
Functional Requirements
Requirement	Status
Multi-class classification	✅ Implemented
Image preprocessing / normalization	✅ Implemented
Model evaluation	✅ Implemented
Prediction on unseen images	✅ Implemented
Single-image upload	✅ Implemented
Bulk ZIP upload	✅ Implemented
Confidence score	✅ Implemented
Bulk result table	✅ Implemented
Configurable confidence threshold	✅ Implemented
Low-confidence review handling	✅ Implemented
Unknown / unclassified handling	✅ Implemented


Engineering Requirements
Requirement	Status
PEP 8 / automated linting	✅
Type checking	✅
Unit testing	✅
Coverage validation	✅
No hardcoded secrets	✅
Docker	✅
Git branching	✅
Pull Requests	✅
CI pipeline	✅
CD pipeline	✅
Azure deployment	✅
Managed Identity / RBAC	✅
Azure logging	✅


BRD Items Requiring Separate Consideration
Azure Blob Storage Ingestion
Azure Blob Storage has been provisioned as part of the project Azure environment.
However, the current end-user prediction flow is:
User Upload
    ↓
Streamlit
    ↓
FastAPI
    ↓
Model

The application does not currently use Azure Blob Storage as an intermediate storage location for every user-uploaded prediction image.
Therefore, this README does not claim direct Blob-based prediction ingestion as an implemented feature.
Grad-CAM
Grad-CAM/explainability visualization was identified as an optional Could Have requirement in the BRD and is not included in the current production implementation.

🔮 Future Improvements
Possible future enhancements include:
1. Dedicated Azure Blob Storage ingestion pipeline.
2. Dedicated labeled Unknown/OOD evaluation dataset.
3. Confidence calibration.
4. Grad-CAM explainability visualization.
5. Additional monitoring dashboards.
6. Automated model versioning.
7. Experiment tracking.
8. Further inference optimization.
9. Expanded scene categories.
10. Larger and more diverse evaluation datasets.

🎯 Project Outcome
The project demonstrates an end-to-end machine learning engineering workflow:
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

The resulting application provides a practical interface for satellite scene classification while demonstrating machine learning, backend development, frontend development, containerization, cloud deployment, CI/CD, testing, and monitoring.

👤 Project Author
Tanvi Jivatode

Project:
Satellite Image Scene Classification

📌 Project Status
Production Internship Submission
The current project includes:
- ✅ Trained classification model
- ✅ Model evaluation
- ✅ 95.25% known-class test accuracy
- ✅ Single-image prediction
- ✅ Bulk ZIP prediction
- ✅ Confidence scoring
- ✅ Configurable review threshold
- ✅ Unknown / unclassified handling
- ✅ CSV export
- ✅ Premium Streamlit UI
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
- ✅ Production file-upload session issue resolved

📄 License
This project was developed as part of an internship project and is intended for educational, demonstration, and internship evaluation purposes.


### Important

This is the **complete README**, not a partial draft.

And I intentionally made the BRD section honest: **we will not write that Blob ingestion is implemented when the current application doesn't actually do it.** That protects you if your lead asks about it.

For the **README itself, don't make any more code changes**. Paste this entire content into `README.md`.

After that, our documentation work can move to the next deliverable **without touching the working application**.
