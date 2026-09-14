# 📱 Smartphone Addiction Detection & Risk Predictor

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartphone-addiction-prediction-agxpqdnmzvtanfhugz7ghk.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

An end-to-end Machine Learning web application designed to predict student smartphone addiction risk levels (Low, Medium, High) based on behavioral, emotional, and academic indicators.

---

## 🚀 Live Application
👉 **[Click here to open the Live Demo](https://smartphone-addiction-prediction-agxpqdnmzvtanfhugz7ghk.streamlit.app/)**

---

## 📁 Project Structure

`
smartphone-addiction-detection-using-ml/
├── app.py                        # Streamlit web application
├── train.py                      # Model training & auto-selection pipeline
├── requirements.txt              # Python dependencies
├── data/
│   └── phone_addiction.csv       # Student habits dataset (3,000 samples)
└── model/
    └── best_model.pkl            # Final pickled pipeline (Preprocessors + SMOTE + LogisticRegression)
`

---

## 📊 Dataset Analysis & Performance Summary

* **Overall Accuracy**: **98.33%**
* **F1 Macro Score**: **0.9773**
* **Weighted F1 Score**: **0.9837**

### Model Evaluation Outcomes

| Model Configuration | Cross-Validation F1 Macro | Accuracy | Passes Sanity Check? | Selected? |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (SMOTE + Feature Ratios)** | **0.8958** | **98.33%** | **Yes** | **Yes (Best & Balanced)** |
| Random Forest (SMOTE) | 0.7546 | 94.83% | No | No |

---

## ⚙️ Features Used

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| **Daily Usage Hours** | Numeric | Total hours spent on phone daily |
| **Sleep Hours** | Numeric | Average hours of sleep per night |
| **Time on Social Media** | Numeric | Hours spent on social media |
| **Time on Gaming** | Numeric | Hours spent on gaming |
| **Apps Used Daily** | Numeric | Total apps opened per day |
| **Phone Checks Per Day** | Numeric | Number of times student unlocks phone |
| **Usage to Sleep Ratio** | Engineered | Daily_Usage_Hours / (Sleep_Hours + 0.1) |
| **Social Plus Gaming** | Engineered | Time_on_Social_Media + Time_on_Gaming |
| **Screen Per Check** | Engineered | Daily_Usage_Hours / (Phone_Checks_Per_Day + 1.0) |
| **Anxiety Level** | Numeric | Self-reported anxiety score (0-10) |
| **Academic Performance** | Numeric | Latest GPA / Grade percentage (0-100) |
| **Age** | Numeric | Student's age |
| **Gender** | Categorical | Male / Female / Other |

---

## 💻 Run Locally

`ash
# 1. Clone repository
git clone https://github.com/Maliniuppalapati/Smartphone-Addiction-Prediction.git
cd Smartphone-Addiction-Prediction

# 2. Virtual environment & requirements
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

# 3. Train model & run Streamlit
python train.py
streamlit run app.py
`

---

## ✍️ Author

**Malini Uppalapati**
* [GitHub Profile](https://github.com/maliniuppalapati)
