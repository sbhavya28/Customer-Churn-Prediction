# 📊 Customer Churn Prediction using XGBoost
redicting customer churn using machine learning (XGBoost). This project analyzes historical customer data (demographics, account activity, financials) to identify users likely to leave a service. Includes data preprocessing, model training, evaluation (ROC-AUC, classification report), and feature importance insights. Dataset: Kaggle Bank Churn.


## 🧠 Task Objective

The primary goal of this project is to build a machine learning model that predicts whether a customer is likely to discontinue a subscription-based service (churn). By analyzing historical customer data, we aim to:
- Accurately predict churn.
- Understand the key factors that influence customer churn.
- Provide actionable insights for customer retention strategies.

---

## 📁 Dataset

Dataset used: [Bank Customer Churn Prediction - Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)

It contains customer information such as:
- Demographics (age, gender, geography)
- Financial information (credit score, balance, salary)
- Service usage metrics (tenure, number of products, active status)

---

## ⚙️ Steps to Run the Project

### 1. Clone the repository or download the files
```bash
git clone https://github.com/sbhavya28/churn-prediction-xgboost.git
cd churn-prediction-xgboost
```
### 2. Set Up Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate      # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Dataset
Download Churn_Modelling.csv from the Kaggle link and place it in the root project directory.

### 5. Run the Code
You can either:

Open and run churn_prediction.ipynb in Jupyter Notebook

Or execute the script file:
```bash
python churn_prediction.py
```

🧪 Model & Evaluation
Model Used: XGBoost Classifier

Metrics:

Accuracy

Precision, Recall, F1-Score

ROC-AUC Score

Model Outputs:

Confusion Matrix

Feature Importance Plot

Churn probability predictions

📌 Feature Importance
XGBoost allows us to visualize which features contributed most to the decision-making. Example important features:

Age

Balance

Tenure

IsActiveMember

Geography

📈 Sample Evaluation Output

```yaml

              precision    recall  f1-score   support

           0       0.87      0.94      0.90      1595
           1       0.77      0.58      0.66       405

    accuracy                           0.85      2000
   macro avg       0.82      0.76      0.78      2000
weighted avg       0.85      0.85      0.85      2000



ROC-AUC Score: 0.87
```
🧾 Requirements
Python 3.7+

pandas

numpy

matplotlib

seaborn

scikit-learn

xgboost

jupyter (optional)

Install all packages:
```bash
pip install -r requirements.txt
```
📬 Contact
Developed by: Bhavya Shukla

Feel free to connect or reach out for feedback or collaboration opportunities!

📧 shukla.bhavya28@gmail.com
🔗 www.linkedin.com/in/bhavya-shukla-6782a8268


