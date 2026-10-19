
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "iris_model.joblib"

# Load the saved model
model = joblib.load(MODEL_PATH)

# Enter the flower's four measurements in centimeters
new_flower = pd.DataFrame([{
    "SepalLengthCm": 5.1,
    "SepalWidthCm": 3.5,
    "PetalLengthCm": 1.4,
    "PetalWidthCm": 0.2
}])

# Predict the species
prediction = model.predict(new_flower)[0]

print("Flower measurements:")
print(new_flower.to_string(index=False))
print(f"\nPredicted species: {prediction}")
