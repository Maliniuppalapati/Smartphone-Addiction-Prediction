import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "phone_addiction.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

def generate_synthetic_data(num_samples=1000):
    """Generates realistic synthetic smartphone addiction data if dataset file does not exist."""
    np.random.seed(42)
    
    daily_usage = np.random.uniform(0.5, 12.0, num_samples)
    sleep_hours = np.random.uniform(4.0, 9.5, num_samples)
    anxiety_level = np.random.randint(0, 11, num_samples)
    academic_perf = np.random.randint(40, 101, num_samples)
    age = np.random.randint(12, 25, num_samples)
    gender = np.random.choice(["Male", "Female", "Other"], size=num_samples, p=[0.48, 0.48, 0.04])
    apps_used = np.random.randint(3, 40, num_samples)
    social_media = np.clip(daily_usage * np.random.uniform(0.2, 0.6, num_samples), 0, 10)
    gaming = np.clip(daily_usage * np.random.uniform(0.1, 0.4, num_samples), 0, 8)
    phone_checks = np.random.randint(15, 180, num_samples)
    
    # Calculate synthetic addiction score (0 to 10)
    addiction_score = (
        0.35 * daily_usage 
        - 0.25 * sleep_hours 
        + 0.3 * anxiety_level 
        - 0.03 * (academic_perf - 50) 
        + 0.2 * social_media 
        + 0.15 * gaming 
        + 0.015 * phone_checks
    )
    # Normalize score between 0 and 10
    min_s, max_s = addiction_score.min(), addiction_score.max()
    addiction_level = 10 * (addiction_score - min_s) / (max_s - min_s)
    
    df_synthetic = pd.DataFrame({
        "Daily_Usage_Hours": np.round(daily_usage, 2),
        "Sleep_Hours": np.round(sleep_hours, 2),
        "Anxiety_Level": anxiety_level,
        "Academic_Performance": academic_perf,
        "Age": age,
        "Gender": gender,
        "Apps_Used_Daily": apps_used,
        "Time_on_Social_Media": np.round(social_media, 2),
        "Time_on_Gaming": np.round(gaming, 2),
        "Phone_Checks_Per_Day": phone_checks,
        "Addiction_Level": np.round(addiction_level, 2)
    })
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_synthetic.to_csv(DATA_PATH, index=False)
    print(f"Generated synthetic dataset with {num_samples} samples at {DATA_PATH}")
    return df_synthetic

def train_and_save_model():
    # Load dataset or generate if missing
    if not DATA_PATH.exists():
        df = generate_synthetic_data()
    else:
        df = pd.read_csv(DATA_PATH)

    # Target classification
    df["Addiction_Level_Class"] = pd.cut(
        df["Addiction_Level"],
        bins=[-0.001, 4, 7, 10],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )
    df.dropna(subset=["Addiction_Level_Class"], inplace=True)
    print("Target Class Counts:")
    print(df["Addiction_Level_Class"].value_counts())

    # Feature Engineering
    df["Usage_to_Sleep_Ratio"] = df["Daily_Usage_Hours"] / (df["Sleep_Hours"] + 0.1)
    df["Social_Plus_Gaming"] = df["Time_on_Social_Media"] + df["Time_on_Gaming"]
    df["Screen_Per_Check"] = df["Daily_Usage_Hours"] / (df["Phone_Checks_Per_Day"] + 1.0)

    FEATURES = [
        "Daily_Usage_Hours",
        "Sleep_Hours",
        "Anxiety_Level",
        "Academic_Performance",
        "Age",
        "Gender",
        "Apps_Used_Daily",
        "Time_on_Social_Media",
        "Time_on_Gaming",
        "Phone_Checks_Per_Day",
        "Usage_to_Sleep_Ratio",
        "Social_Plus_Gaming",
        "Screen_Per_Check"
    ]

    X = df[FEATURES].copy()
    y = df["Addiction_Level_Class"]

    numeric_features = [c for c in FEATURES if c != "Gender"]
    categorical_features = ["Gender"]

    # Preprocessor
    numeric_transform = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transform = ImbPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_transform, numeric_features),
        ("cat", categorical_transform, categorical_features)
    ])

    # Split data first to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Pipelines with SMOTE for balance
    pipelines = {
        "LogisticRegression_SMOTE": ImbPipeline([
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("clf", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
        ]),
        "RandomForest_SMOTE": ImbPipeline([
            ("preprocess", preprocess),
            ("smote", SMOTE(random_state=42, k_neighbors=3)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
        ])
    }

    def passes_sanity_check(pipeline):
        test_df = pd.DataFrame([{
            "Daily_Usage_Hours": 0.0,
            "Sleep_Hours": 8.0,
            "Anxiety_Level": 0,
            "Academic_Performance": 100,
            "Age": 20,
            "Gender": "Male",
            "Apps_Used_Daily": 1,
            "Time_on_Social_Media": 0.0,
            "Time_on_Gaming": 0.0,
            "Phone_Checks_Per_Day": 5,
            "Usage_to_Sleep_Ratio": 0.0 / 8.1,
            "Social_Plus_Gaming": 0.0,
            "Screen_Per_Check": 0.0 / 6.0
        }])
        try:
            pred = pipeline.predict(test_df)[0]
            return pred == "Low"
        except Exception as e:
            print("Sanity check exception:", e)
            return False

    # Cross-Validation & Model Selection
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model_name = None
    best_score = -1
    best_pipeline = None

    print("\n==== MODEL SELECTION AND EVALUATION ====")
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        passed = passes_sanity_check(pipe)
        
        cv_scores = cross_val_score(pipe, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=-1)
        mean_f1_macro = np.mean(cv_scores)
        
        print(f"\nModel: {name}")
        print(f"  Cross-Validation F1 Macro: {mean_f1_macro:.4f}")
        print(f"  Passes Low-Addiction Sanity Check: {passed}")
        
        if passed and mean_f1_macro > best_score:
            best_score = mean_f1_macro
            best_model_name = name
            best_pipeline = pipe

    if best_pipeline is None:
        print("\nFallback to LogisticRegression_SMOTE")
        best_model_name = "LogisticRegression_SMOTE"
        best_pipeline = pipelines["LogisticRegression_SMOTE"]
        best_pipeline.fit(X_train, y_train)

    # Final evaluation on held-out test set
    y_pred = best_pipeline.predict(X_test)

    print("\n==== FINAL SELECTED MODEL PERFORMANCE ====")
    print(f"Selected model: {best_model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Macro:", f1_score(y_test, y_pred, average="macro"))
    print("F1 Weighted:", f1_score(y_test, y_pred, average="weighted"))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    # Save pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nModel successfully saved at: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
