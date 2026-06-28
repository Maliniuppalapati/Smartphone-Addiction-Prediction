import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "phone_addiction.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

# Features to use
FEATURES = [
    "Daily_Usage_Hours",
    "Sleep_Hours",
    "Anxiety_Level",
    "Academic_Performance",
    "Age",
    "Gender"
]

# Load dataset
df = pd.read_csv(DATA_PATH)

# Create target classes
df["Addiction_Level_Class"] = pd.cut(
    df["Addiction_Level"],
    bins=[-0.001, 4, 7, 10],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

df.dropna(subset=["Addiction_Level_Class"], inplace=True)
print(df["Addiction_Level_Class"].value_counts())

X = df[FEATURES].copy()
y = df["Addiction_Level_Class"]

# Detect numeric & categorical
numeric_features = ["Daily_Usage_Hours", "Sleep_Hours", "Anxiety_Level", "Academic_Performance", "Age"]
categorical_features = ["Gender"]

# Pipelines
numeric_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transform = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", numeric_transform, numeric_features),
    ("cat", categorical_transform, categorical_features)
])

# Models
models = {
    "LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

pipelines = {name: Pipeline([("preprocess", preprocess), ("clf", model)])
             for name, model in models.items()}

# Split data first to evaluate model selection on the training set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# Sanity check function to prevent biased predictions on low-addiction profiles
def passes_sanity_check(pipeline):
    test_df = pd.DataFrame([{
        "Daily_Usage_Hours": 0.0,
        "Sleep_Hours": 8.0,
        "Anxiety_Level": 0,
        "Academic_Performance": 100,
        "Age": 20,
        "Gender": "Male"
    }])
    try:
        pred = pipeline.predict(test_df)[0]
        return pred == "Low"
    except Exception:
        return False

# Evaluate each model on cross-validation and verify sanity check
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model_name = None
best_score = -1
best_pipeline = None

print("\n==== MODEL SELECTION AND EVALUATION ====")
for name, pipe in pipelines.items():
    # Fit on training data first to test the sanity check
    pipe.fit(X_train, y_train)
    passed = passes_sanity_check(pipe)
    
    # Calculate cross-validation F1 Macro (more robust for imbalanced classes)
    cv_scores = cross_val_score(pipe, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1)
    mean_f1_macro = np.mean(cv_scores)
    
    print(f"\nModel: {name}")
    print(f"  Cross-Validation F1 Macro: {mean_f1_macro:.4f}")
    print(f"  Passes Low-Addiction Sanity Check (0h usage, 8h sleep -> 'Low'): {passed}")
    
    # Select the model with the highest F1 Macro that passes the sanity check
    if passed and mean_f1_macro > best_score:
        best_score = mean_f1_macro
        best_model_name = name
        best_pipeline = pipe

# Fallback in case no model passes the sanity check (highly unlikely)
if best_pipeline is None:
    print("\nWARNING: No model passed the sanity check! Falling back to LogisticRegression.")
    best_model_name = "LogisticRegression"
    best_pipeline = pipelines["LogisticRegression"]
    best_pipeline.fit(X_train, y_train)

# Final evaluation of the selected model on the test set
y_pred = best_pipeline.predict(X_test)

print("\n==== FINAL SELECTED MODEL PERFORMANCE ====")
print(f"Selected model: {best_model_name}")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))
print("F1 Weighted:", f1_score(y_test, y_pred, average="weighted"))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save the selected model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_pipeline, MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")