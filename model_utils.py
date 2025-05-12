# model_utils.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(df[['Time', 'Amount']])
    scaled_df = pd.DataFrame(scaled_features, columns=['scaled_time', 'scaled_amount'])
    df = pd.concat([df, scaled_df], axis=1)
    df.drop(['Time', 'Amount'], axis=1, inplace=True)

    fraud = df[df['Class'] == 1]
    non_fraud = df[df['Class'] == 0].sample(n=len(fraud), random_state=42)
    balanced_data = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)

    X = balanced_data.drop('Class', axis=1)
    y = balanced_data['Class']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_model(name):
    if name == "logistic":
        return LogisticRegression()
    elif name == "tree":
        return DecisionTreeClassifier(random_state=42)
    elif name == "forest":
        return RandomForestClassifier(n_estimators=100, random_state=42)

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True)
    matrix = confusion_matrix(y_test, predictions).tolist()
    accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
    return report, matrix, accuracy
