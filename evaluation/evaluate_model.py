"""
Evaluate the trained Satellite Scene Classification model.

Uses the same preprocessing and prediction logic as the FastAPI backend.
Evaluates images in dataset/test.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the actual backend model and prediction logic
from backend.main import (
    CLASS_NAMES,
    KNOWN_CLASS_NAMES,
    load_and_prepare,
    predict_array,
)

# Dataset location
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map dataset folder names to model class names
CLASS_MAPPING  = {
    "forest": "Forest",
    "lake": "SeaLake",
    "desert": "Desert",
    "cloud": "Cloudy",
    "unknown": "Unknown",
}

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def evaluate_model():
    y_true = []
    y_pred = []
    results = []

    print("=" * 60)
    print("SATELLITE SCENE CLASSIFICATION - MODEL EVALUATION")
    print("=" * 60)

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

    print(f"Test dataset: {TEST_DIR}")
    print(f"Model classes: {CLASS_NAMES}")

    # Process each class folder
    for folder in sorted(TEST_DIR.iterdir()):
        if not folder.is_dir():
            continue

        folder_name = folder.name.lower()

        if folder_name not in CLASS_MAPPING:
            print(f"Skipping unsupported class: {folder.name}")
            continue

        actual_class = CLASS_MAPPING[folder_name]

        if actual_class not in CLASS_NAMES:
            print(f"Skipping class not supported by model: {actual_class}")
            continue

        image_files = sorted(
            file
            for file in folder.rglob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\nEvaluating {actual_class}: {len(image_files)} images")

        for image_path in image_files:
            try:
                image_bytes = image_path.read_bytes()

                # Same preprocessing as the backend
                arr = load_and_prepare(image_bytes)

                # Same prediction logic as the backend
                prediction = predict_array(arr, filename=image_path.name)

                # A rejected prediction counts as Unknown
                # for this operational evaluation.
                predicted_class = (
                    prediction.predicted_class
                    if prediction.status == "KNOWN"
                    else "Unknown"
                )

                y_true.append(actual_class)
                y_pred.append(predicted_class)

                results.append(
                    {
                        "filename": str(image_path.relative_to(TEST_DIR)),
                        "actual_class": actual_class,
                        "predicted_class": prediction.predicted_class,
                        "final_result": prediction.final_result,
                        "status": prediction.status,
                        "confidence": prediction.confidence,
                        "p_unknown": prediction.p_unknown,
                        "evaluation_prediction": predicted_class,
                        "correct": predicted_class == actual_class,
                    }
                )

            except (OSError, ValueError, RuntimeError) as error:
                print(f"Error processing {image_path.name}: {error}")

    if not y_true:
        raise ValueError("No images were successfully evaluated.")

    # Save individual predictions
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "predictions.csv"
    results_df.to_csv(results_path, index=False)

    # Evaluation labels include Unknown to account for rejected images
    evaluation_labels = KNOWN_CLASS_NAMES + ["Unknown"]

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"Total evaluated images: {len(y_true)}")
    print(f"Correct predictions: {(np.array(y_true) == np.array(y_pred)).sum()}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\nClassification Report:")
    report = classification_report(
        y_true,
        y_pred,
        labels=evaluation_labels,
        zero_division=0,
    )
    print(report)

    # Save classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=evaluation_labels,
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(OUTPUT_DIR / "classification_report.csv")

    # Generate and save confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=evaluation_labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=evaluation_labels,
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
    )
    plt.title("Satellite Scene Classification - Confusion Matrix")
    plt.tight_layout()

    confusion_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(confusion_path, dpi=300)
    plt.close(fig)

    # Save summary
    summary = {
        "total_images": len(y_true),
        "accuracy": accuracy,
        "known_predictions": int((results_df["status"] == "KNOWN").sum()),
        "review_predictions": int((results_df["status"] == "REVIEW").sum()),
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR / "evaluation_summary.csv",
        index=False,
    )

    print("\nFiles saved:")
    print(f"Predictions: {results_path}")
    print(f"Classification report: {OUTPUT_DIR / 'classification_report.csv'}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Summary: {OUTPUT_DIR / 'evaluation_summary.csv'}")

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    evaluate_model()
