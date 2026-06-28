# 📱 Smartphone Addiction Detection & Risk Predictor

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartphone-addiction-prediction-agxpqdnmzvtanfhugz7ghk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

An end-to-end Machine Learning web application designed to predict student smartphone addiction risk levels (Low, Medium, High) based on behavioral, emotional, and academic indicators. 

---

## 🌐 Live Application
🔗 **[Click here to open the Live Demo](https://smartphone-addiction-prediction-agxpqdnmzvtanfhugz7ghk.streamlit.app/)**

---

## 📁 Project Structure

```
smartphone-addiction-detection-using-ml/
├── app.py                        # Streamlit web application
├── train.py                      # Model training & auto-selection pipeline
├── requirements.txt              # Python dependencies
├── data/
│   └── phone_addiction.csv       # Student habits dataset (3,000 samples)
└── model/
    └── best_model.pkl            # Final pickled pipeline (Preprocessors + LogisticRegression)
```

---

## 📊 Dataset Analysis & Statistics

The dataset (`data/phone_addiction.csv`) consists of **3,000 entries** and **22 columns**, tracking various student demographics and screen habit parameters. 

### 1. The Target Variable & Severe Class Imbalance
The continuous addiction score (range 1–10) was grouped into 3 risk tiers (`Low`, `Medium`, `High`). This created a highly imbalanced classification task:
* **High Risk** (Score 7.0 – 10.0): **2,554 samples** (85.1%)
* **Medium Risk** (Score 4.0 – 7.0): **404 samples** (13.5%)
* **Low Risk** (Score 0.0 – 4.0): **42 samples** (1.4%)

### 2. Feature Correlations
We analyzed the correlation of different features against the original continuous addiction score:
* **Daily_Usage_Hours**: `+0.601` (Strong positive correlation)
* **Apps_Used_Daily**: `+0.319` (Moderate positive correlation)
* **Time_on_Social_Media**: `+0.307` (Moderate positive correlation)
* **Sleep_Hours**: `-0.217` (Negative correlation — less sleep corresponds to higher addiction)
* **Anxiety_Level**: `+0.016` (Low direct linear correlation)
* **Academic_Performance**: `+0.012` (Low direct linear correlation)

---

## 🛠️ Bias Correction & Model Selection

### The Problem with the Original Model
Initially, the pipeline was trained using standard Random Forest and evaluated using `f1_weighted`. 
* Because `f1_weighted` is dominated by the majority class (`High`), the model ignored the `Low` class entirely (yielding **0.00 recall**).
* This led to highly biased, incorrect predictions. For instance, entering `0.0` daily screen hours and `8.0` sleep hours still predicted a "Medium" or "High" addiction level.

### How We Solved It
We updated the model selection workflow in `train.py` to ensure fairness and logical consistency:
1. **Balanced Class Weights**: Enabled `class_weight="balanced"` on all estimators to penalize errors in minority classes.
2. **Macro F1 Selection**: Shifted the cross-validation optimization metric to `f1_macro` so that the `Low` class receives equal importance.
3. **Behavioral Sanity Check**: Implemented a validation check inside the training loop. Any candidate model must correctly predict **"Low"** for a student with `0.0` screen hours and `8.0` sleep hours to be chosen.

### Model Evaluation Outcomes

| Model Configuration | Cross-Validation F1 Macro | Passes Sanity Check? | Selected? |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **0.4843** | **Yes** | **Yes (Best & Logical)** |
| Random Forest (Balanced) | 0.4458 | No (Overfits to local noise) | No |

*Why Logistic Regression won:* Linear decision boundaries generalize much better to extreme values (like 0 usage hours), enforcing a logical, monotonic trend that prevents local overfitting.

---

## 🎯 Input Features Used

| Feature Name | Input Type | Range | Description |
| :--- | :--- | :--- | :--- |
| **Daily Usage Hours** | Numeric | 0.0 – 24.0 | Total hours spent on phone daily |
| **Sleep Hours** | Numeric | 0.0 – 15.0 | Average hours of sleep per night |
| **Anxiety Level** | Numeric | 0 – 10 | Self-reported anxiety score |
| **Academic Performance** | Numeric | 0 – 100 | Latest GPA / Grade percentage |
| **Age** | Numeric | 5 – 25 | Student's age |
| **Gender** | Categorical | Male / Female / Other | Demographic gender classification |

---

## 🚀 Run Locally

Ensure you have Python 3.10+ installed.

```bash
# 1. Clone the repository
git clone https://github.com/maliniuppalapati/smartphone-addiction-detection-using-ml-main.git
cd smartphone-addiction-detection-using-ml-main

# 2. Set up a virtual environment and install packages
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# 3. Retrain the model (optional)
python train.py

# 4. Start the Streamlit server
streamlit run app.py
```

---

## 👤 Author

**Malini Uppalapati**
* [GitHub Profile](https://github.com/maliniuppalapati)
