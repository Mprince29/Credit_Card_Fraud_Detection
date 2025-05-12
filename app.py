# app.py

from flask import Flask, render_template, request
from model_utils import load_and_preprocess_data, get_model, evaluate_model

app = Flask(__name__)
DATA_PATH = "creditcard.csv"  # Adjust if needed

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_type = request.form.get("model")
        X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
        model = get_model(model_type)
        report, matrix, accuracy = evaluate_model(model, X_train, X_test, y_train, y_test)
        return render_template("result.html", accuracy=accuracy, matrix=matrix, report=report, model=model_type.title())
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
