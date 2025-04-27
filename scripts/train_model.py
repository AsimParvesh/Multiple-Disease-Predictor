import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from preprocessing import preprocess_parkinsons, preprocess_kidney, preprocess_liver

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Create models folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Load datasets
parkinsons_df = pd.read_csv(os.path.join(DATA_DIR, 'parkinsons.csv'))
kidney_df = pd.read_csv(os.path.join(DATA_DIR, 'kidney.csv'))
liver_df = pd.read_csv(os.path.join(DATA_DIR, 'liver.csv'))

# Preprocess datasets
X_parkinsons, y_parkinsons = preprocess_parkinsons(parkinsons_df)
X_kidney, y_kidney = preprocess_kidney(kidney_df)
X_liver, y_liver = preprocess_liver(liver_df)

# Train and save model function
def train_and_save_model(X, y, model_name, tune=False, apply_smote=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if apply_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    if tune:
        # Hyperparameter tuning with GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                                   param_grid,
                                   cv=3,
                                   n_jobs=-1,
                                   verbose=1)
        grid_search.fit(X_train, y_train)
        clf = grid_search.best_estimator_
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    else:
        # Simple Random Forest
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.2f}")
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    model_path = os.path.join(MODEL_DIR, f"{model_name}_model.pkl")
    joblib.dump(clf, model_path)
    print(f"Saved {model_name} model at {model_path}")

# Train models
train_and_save_model(X_parkinsons, y_parkinsons, "parkinsons")
train_and_save_model(X_kidney, y_kidney, "kidney")
train_and_save_model(X_liver, y_liver, "liver", tune=True, apply_smote=True)
