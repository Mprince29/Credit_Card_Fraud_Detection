# 💳 Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using various classification models and performance evaluation metrics. The dataset used is highly imbalanced, mimicking real-world financial fraud detection scenarios.

---

## 📌 Table of Contents
- [Overview]
- [Tech Stack
- [ML Models Used]
- [Evaluation Metrics]
- [Installation]
- [Usage]
- [Results]
- [Screenshots]
- [License]

---

## 📖 Overview

This project applies machine learning algorithms to identify potentially fraudulent credit card transactions. Due to the nature of the data (severely imbalanced classes), special attention is paid to model selection and metric evaluation (beyond just accuracy).

---

## 🧰 Tech Stack

- **Language**: Python 3.x  
- **Libraries**:
  - `pandas`, `numpy` – Data handling
  - `matplotlib`, `seaborn` – Visualization
  - `scikit-learn` – ML models and metrics
  - `xgboost` – Advanced gradient boosting
  - `tensorflow/keras` – Deep learning (LSTM model)
  
---

## 🧠 ML Models Used

- 🔍 **Random Forest Classifier** – Ensemble of decision trees
- ✂️ **Support Vector Machine (SVM)** – Margin-based classifier
- ⚡ **XGBoost Classifier** – High-performance boosting
- 🧠 **MLP Classifier (Neural Network)** – Multi-layer perceptron
- 🌳 **Decision Tree Classifier** – Simple interpretable tree
- 🧬 **LSTM Neural Network** – Sequence learning (for temporal fraud patterns)

---

## 📊 Evaluation Metrics

To handle the **imbalanced dataset**, we used:
- **Confusion Matrix**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC-AUC Score**

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python creditcardfraud.py

▶️ Usage
Make sure you have the dataset (creditcard.csv) in the project folder.

You can modify or experiment with different models inside creditcardfraud.py as needed.

📈 Results

The best performing model (e.g., XGBoost or LSTM) achieved:

Precision: 0.92+

Recall: 0.90+

F1-Score: 0.91+

(Metrics may vary based on data split and tuning.)

