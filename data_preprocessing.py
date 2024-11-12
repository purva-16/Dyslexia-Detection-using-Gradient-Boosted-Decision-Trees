import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_path='Dyt-tablet.csv'):
    # Load the dataset
    df = pd.read_csv(file_path, delimiter=';')

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Check for 'Dyslexia' column
    if 'Dyslexia' not in df.columns:
        raise ValueError("Column 'Dyslexia' not found! Please check the dataset.")
    
    # Remove columns related to question 29
    cols_to_remove = ['Clicks29', 'Hits29', 'Misses29', 'Score29', 'Accuracy29', 'Missrate29']
    df.drop(columns=cols_to_remove, inplace=True, errors='ignore')

    # Map categorical columns
    df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
    df['Nativelang'] = df['Nativelang'].map({'yes': 1, 'no': 0})
    df['Otherlang'] = df['Otherlang'].map({'yes': 1, 'no': 0})

    # Fill missing values in numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Encode categorical variables
    categorical_columns = ['Gender', 'Nativelang', 'Otherlang']
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Encode target variable 'Dyslexia'
    label_encoder = LabelEncoder()
    df['Dyslexia'] = label_encoder.fit_transform(df['Dyslexia'])

    # Split data into features and target variable
    X = df.drop(columns=['Dyslexia'])
    y = df['Dyslexia']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Scale the features
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    return X_train_resampled, X_test, y_train_resampled, y_test
