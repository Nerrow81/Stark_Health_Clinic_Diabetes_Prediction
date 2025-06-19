# 🩺 Stark Health Clinic – Diabetes Prediction Project

Leveraging machine learning to predict diabetes onset and support early medical interventions.

## 📌 Project Overview

This project was developed for **Stark Health Clinic** with the goal of creating a predictive model to identify patients at risk of developing diabetes. Using real patient data, we built and evaluated multiple machine learning models that enable proactive healthcare decisions and resource optimization.

---

## 🎯 Objectives

- Predict the likelihood of a patient developing diabetes using available health indicators.
- Reduce false negatives in screening processes to improve early diagnosis and care.
- Deliver an explainable, reproducible, and deployable ML pipeline.

---

## 📁 Dataset Summary

- **Total Records:** 96,146  
- **Target Variable:** `diabetes` (binary: 0 = No, 1 = Yes)  
- **Features:** `gender`, `age`, `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `HbA1c_level`, `blood_glucose_level`  
- **Duplicates Removed:** 3,854  
- **Final Rows Used:** 92,292

---

## 🔍 Project Workflow

### 1. Problem Definition
- Identified that Stark Health’s current detection methods lacked precision.
- Focused on minimizing **false negatives** to ensure timely intervention.

### 2. Exploratory Data Analysis (EDA)
- Conducted univariate, bivariate, and multivariate analysis.
- Visualized distributions and relationships using Seaborn and Matplotlib.
- Key insights: High risk linked with hypertension, heart disease, and smoking history.

### 3. Feature Engineering
- Encoded categorical variables using Label Encoding.
- Scaled numerical features using StandardScaler.
- Addressed class imbalance by applying **class weights** in modeling.

### 4. Modeling & Hyperparameter Tuning
- Algorithms tested:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - K-Nearest Neighbors (KNN)  
- Used **RandomizedSearchCV** for hyperparameter tuning.

### 5. Model Evaluation Metrics
| Model               | Accuracy | ROC-AUC | Recall (Diabetes) | False Negatives |
|--------------------|----------|---------|-------------------|------------------|
| Logistic Regression | 88.47%   | 0.9611  | 88%               | 286              |
| Decision Tree       | 95.07%   | 0.8526  | 73%               | 655              |
| Random Forest 🏆     | 93.86%   | 0.9756  | 84%               | 388              |
| KNN                 | 96.10%   | 0.9317  | 61%               | 963              |

> **Best Performing Model**: Random Forest (high recall, AUC, and balanced performance)

---

## ✅ Key Takeaways

- **Recall** was prioritized over accuracy due to the clinical risk of false negatives.
- **Random Forest** showed the best trade-off between predictive power and generalizability.
- Class imbalance was handled effectively using **class weighting** across all models.

---

## 💼 Business Impact

- Early identification of high-risk patients for timely intervention.
- Improved resource allocation and healthcare outcomes.
- Supports data-driven decision-making within Stark Health Clinic.

---

## 🚀 Next Steps

- Integrate the selected model into the clinic’s electronic health system.
- Schedule regular retraining using newly collected patient data.
- Implement model explainability tools (e.g., SHAP, LIME) for clinical transparency.

---

## 🛠️ Tech Stack

- **Languages:** Python  
- **Libraries:** NumPy, Pandas, Seaborn, Matplotlib, Scikit-learn  
- **Platform:** Jupyter Notebook (Anaconda)

---

## 🧾 Author

**Olatunde Emmanuel**  
_Data Scientist _  
[LinkedIn](https://www.linkedin.com/in/) •

---

## 📂 Files in This Repository

- `Stark_Health-clinic_diabetes_pred.ipynb` – Jupyter Notebook with full analysis, modeling, and results
- `requirements.txt` - Having the list of libraries used
- `diabetes_prediction_dataset.csv` - The data set used for this project
- `diabetes_model.pkl` - The best peerforming model
- `Stark Health Clinic Diabetes prediction.pptx` – Slide deck summarizing the project using the STAR format
- `README.md` – Project documentation (this file)

---

## 📝 License

This project is for academic and demonstration purposes only. Do not use it for clinical decisions without further validation.

