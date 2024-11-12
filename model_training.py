from data_preprocessing import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Function to evaluate models
def evaluate_model(y_test, y_pred, model_name):
    print(f"Results for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Dyslexia', 'Dyslexia'], yticklabels=['No Dyslexia', 'Dyslexia'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.show()


def evaluate_and_print_results(y_test, y_pred, model_name):
    evaluate_model(y_test, y_pred, model_name)

# Main function to load data, train models, and evaluate
def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Model 1: Logistic Regression
    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)
    y_pred_log = log_model.predict(X_test)
    evaluate_and_print_results(y_test, y_pred_log, "Logistic Regression")

    # Model 2: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    evaluate_and_print_results(y_test, y_pred_rf, "Random Forest")

    # Model 3: Support Vector Machine (SVM)
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    evaluate_and_print_results(y_test, y_pred_svm, "Support Vector Machine")

    # Model 4: K-Nearest Neighbors (KNN)
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    evaluate_and_print_results(y_test, y_pred_knn, "K-Nearest Neighbors")

    # Model 5: Gradient Boosting Classifier (sklearn)
    gb_model_sklearn = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42)
    gb_model_sklearn.fit(X_train, y_train)
    y_pred_gb = gb_model_sklearn.predict(X_test)
    evaluate_and_print_results(y_test, y_pred_gb, "Gradient Boosting (sklearn)")

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()