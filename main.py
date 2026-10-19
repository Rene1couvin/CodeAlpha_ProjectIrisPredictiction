
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Load the dataset
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "Iris.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nSpecies counts:")
print(df["Species"].value_counts())

# 2. Select input features and target
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = df[features]
y = df["Species"]

print("\nInput features:", features)
print("Target: Species")

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Create and train the model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X_train, y_train)

print("\nModel training completed!")

# 5. Predict species for the test data
y_pred = model.predict(X_test)

# 6. Evaluate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot(cmap="Blues", xticks_rotation=45)
plt.title("Iris Flower Classification")
plt.tight_layout()

# Save the confusion matrix image
OUTPUT_PATH = PROJECT_DIR / "confusion_matrix.png"
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")

print(f"Confusion matrix saved to: {OUTPUT_PATH}")

plt.show()

# 8. Save the trained model
# Save the trained model
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "iris_model.joblib"

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully to: {MODEL_PATH}")