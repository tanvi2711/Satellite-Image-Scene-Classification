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
