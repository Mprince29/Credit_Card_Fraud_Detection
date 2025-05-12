import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the data
file_path = "C:\Users\princ\Desktop\Credit_Card_fraud_detection\creditcard.csv"  # Update this path if necessary
df = pd.read_csv(file_path)

# Function to summarize data
def summarize_data(data):
    print("\nData Head:\n", data.head())
    print("\nData Description:\n", data.describe())
    print("\nClass Distribution:\n", data['Class'].value_counts())

# Function to visualize data
def visualize_data(data):
    sns.countplot(x='Class', data=data)
    plt.title('Class Distribution')
    plt.xlabel('Transaction Class')
    plt.ylabel('Count')
    plt.show()

# Function for preprocessing data
def preprocess_data(data):
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(data[['Time', 'Amount']])
    scaled_df = pd.DataFrame(scaled_features, columns=['scaled_time', 'scaled_amount'])
    data = pd.concat([data, scaled_df], axis=1)
    data.drop(['Time', 'Amount'], axis=1, inplace=True)

    # Balancing the dataset
    fraud = data[data['Class'] == 1]
    non_fraud = data[data['Class'] == 0].sample(n=len(fraud), random_state=42)
    balanced_data = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=42)
    return balanced_data

# Function to train and evaluate a model
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nAccuracy Score:", round(accuracy_score(y_test, predictions) * 100, 2))

# Main interactive menu
def main():
    while True:
        print("\nMenu:")
        print("1. Summarize Data")
        print("2. Visualize Data")
        print("3. Preprocess Data and Train Models")
        print("4. Exit")

        # Accept user input
        try:
            choice = input("Enter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n--- Summarizing Data ---")
                summarize_data(df)

            elif choice == "2":
                print("\n--- Visualizing Data ---")
                visualize_data(df)

            elif choice == "3":
                print("\n--- Preprocessing Data and Training Models ---")
                processed_df = preprocess_data(df)
                X = processed_df.drop('Class', axis=1)
                y = processed_df['Class']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                print("\nChoose a Model:")
                print("1. Logistic Regression")
                print("2. Decision Tree")
                print("3. Random Forest")
                model_choice = input("Enter your choice (1-3): ").strip()

                if model_choice == "1":
                    print("\nTraining Logistic Regression...")
                    model = LogisticRegression()
                    train_and_evaluate(model, X_train, X_test, y_train, y_test)
                elif model_choice == "2":
                    print("\nTraining Decision Tree...")
                    model = DecisionTreeClassifier(random_state=42)
                    train_and_evaluate(model, X_train, X_test, y_train, y_test)
                elif model_choice == "3":
                    print("\nTraining Random Forest...")
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    train_and_evaluate(model, X_train, X_test, y_train, y_test)
                else:
                    print("Invalid model choice! Please select a valid option.")

            elif choice == "4":
                print("\nExiting the program. Goodbye!")
                break

            else:
                print("\nInvalid choice! Please enter a number between 1 and 4.")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue

# Entry point
if __name__ == "__main__":
    main()
