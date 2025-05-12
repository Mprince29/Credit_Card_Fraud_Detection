# 💳 Credit Card Fraud Detection System

This project is a complete **Credit Card Fraud Detection System** that uses machine learning to identify fraudulent transactions. It provides both:

- ✅ A **Command-Line Interface (CLI)** for exploring data and training models.
- 🌐 A **Flask Web App** where users can interactively select models and view performance metrics.

---

## 🚀 Features

- 📊 Data exploration and visualization (EDA)
- ⚙️ Preprocessing using `RobustScaler`
- ⚖️ Balancing imbalanced datasets
- 🧠 Model training with:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- 🖥️ Web interface with Flask
- 📈 Output includes Accuracy, Confusion Matrix, and Classification Report

---

## 📁 Project Structure

├── app.py # Flask web application

├── creditcardfrauddetection.py # CLI for data exploration and training

├── model_utils.py # Data loading, preprocessing, model evaluation

├── templates/

│ ├── index.html # Model selection form

│ └── result.html # Results display

├── static/ # (Optional) Static assets (CSS, JS)

└── creditcard.csv # Dataset (add this manually)


2. Install Dependencies

Make sure you have Python 3.7+ and run:

pip install -r requirements.txt

🧪 How to Use

🔧 Option 1: CLI Version

Run:

python creditcardfrauddetection.py

You will be able to:

View data summary

Visualize class distribution

Preprocess and train models

🌐 Option 2: Web App

Ensure creditcard.csv is in the root directory.

Run the Flask app:

python app.py

Open your browser at http://127.0.0.1:5000/

Choose a model and view the evaluation results.

🧠 Models Supported

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

📊 Metrics Displayed

Accuracy Score

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

📂 Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset.

File name must be: creditcard.csv

Place it in the project root directory

📄 Requirements

Sample requirements.txt:

flask

pandas

seaborn

matplotlib

scikit-learn

pip install -r requirements.txt


👨‍💻 Author

Master Prince
Final Year B.Tech CSE (AI) Student
Full Stack Developer | AI Enthusiast


