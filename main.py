import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import numpy as np
from gradient_boosting import GradientBoostingModel  # Importing the custom algorithm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from a_plot_data import evaluate_model

# Load the dataset
df = pd.read_csv('Dyt-tablet.csv', delimiter=';')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Check for 'Dyslexia' column
if 'Dyslexia' not in df.columns:
    raise ValueError("Column 'Dyslexia' not found! Please check the dataset.")
else:
    # Remove columns related to question 29
    cols_to_remove = ['Clicks29', 'Hits29', 'Misses29', 'Score29', 'Accuracy29', 'Missrate29']
    df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

    # Assuming df is your DataFrame
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    df['Nativelang'] = df['Nativelang'].map({'yes': 1, 'no': 0})
    df['Otherlang'] = df['Otherlang'].map({'yes': 1, 'no': 0})

    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode categorical columns
    categorical_columns = ['Gender', 'Nativelang', 'Otherlang']  # Adjust as needed
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Encode target variable 'Dyslexia'
    label_encoder = LabelEncoder()
    df['Dyslexia'] = label_encoder.fit_transform(df['Dyslexia'])
    # Split data into features and target variable
    X = df.drop(columns=['Dyslexia'])
    y = df['Dyslexia']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE (Optional, if needed for balancing)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # Initialize and train the custom Gradient Boosting Model
    gb_model = GradientBoostingModel(learning_rate=0.1, n_estimators=100, max_depth=3)  # Your custom model
    gb_model.fit(X_train_resampled, y_train_resampled)

    # Make predictions on the test set
    y_pred = gb_model.predict(X_test)

    evaluate_model(y_test, y_pred)

