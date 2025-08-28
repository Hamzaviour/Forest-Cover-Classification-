"""
Model Training Script for Forest Cover Classification
Created by Hamza Younas
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

def load_and_preprocess_data():
    """Load and preprocess the forest cover dataset"""
    print("Loading dataset...")
    
    column_names = [
        'Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3',
        'Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6',
        'Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13',
        'Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20',
        'Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27',
        'Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34',
        'Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Cover_Type'
    ]
    
    df = pd.read_csv('covtype.csv', header=None, names=column_names)
    
    # Convert data types and clean
    df['Cover_Type_numeric'] = pd.to_numeric(df['Cover_Type'], errors='coerce')
    df_clean = df[df['Cover_Type_numeric'].notna()].copy()
    
    # Prepare features and target
    X = df_clean.drop(['Cover_Type', 'Cover_Type_numeric'], axis=1)
    y = df_clean['Cover_Type_numeric'].astype(int) - 1  # convert 1-7 -> 0-6
    
    # Convert all features to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median())
    
    print(f"Dataset shape: {df_clean.shape}")
    print(f"Features: {X.shape}, Target classes: {y.nunique()}")
    
    return X, y

def train_models(X, y):
    """Train all three models and return them with their performance metrics"""
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    models = {}
    metrics = {}
    
    # Random Forest
    print("Training Random Forest...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)
    
    models['Random Forest'] = rf_clf
    metrics['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
        'classification_report': classification_report(y_test, y_pred_rf, output_dict=True),
        'feature_importance': dict(zip(X.columns, rf_clf.feature_importances_))
    }
    
    # XGBoost
    print("Training XGBoost...")
    xgb_clf = XGBClassifier(
        objective='multi:softmax',
        num_class=int(y.nunique()),
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    
    models['XGBoost'] = xgb_clf
    metrics['XGBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'confusion_matrix': confusion_matrix(y_test, y_pred_xgb),
        'classification_report': classification_report(y_test, y_pred_xgb, output_dict=True),
        'feature_importance': dict(zip(X.columns, xgb_clf.feature_importances_))
    }
    
    # Decision Tree
    print("Training Decision Tree...")
    dt_clf = DecisionTreeClassifier(
        random_state=42,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5
    )
    dt_clf.fit(X_train, y_train)
    y_pred_dt = dt_clf.predict(X_test)
    
    models['Decision Tree'] = dt_clf
    metrics['Decision Tree'] = {
        'accuracy': accuracy_score(y_test, y_pred_dt),
        'confusion_matrix': confusion_matrix(y_test, y_pred_dt),
        'classification_report': classification_report(y_test, y_pred_dt, output_dict=True),
        'feature_importance': dict(zip(X.columns, dt_clf.feature_importances_))
    }
    
    return models, metrics, X_test, y_test

def save_models(models, metrics, X_test, y_test):
    """Save models and metrics to pickle files"""
    print("Saving models and metrics...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save models
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(models['Random Forest'], f)
    
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(models['XGBoost'], f)
    
    with open('models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(models['Decision Tree'], f)
    
    # Save metrics
    with open('models/model_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)
    
    # Save test data for predictions
    with open('models/test_data.pkl', 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)
    
    print("Models and metrics saved successfully!")

if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data()
    
    # Train models
    models, metrics, X_test, y_test = train_models(X, y)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    for model_name, model_metrics in metrics.items():
        accuracy = model_metrics['accuracy']
        print(f"{model_name:<15}: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*60)
    
    # Save models
    save_models(models, metrics, X_test, y_test)
