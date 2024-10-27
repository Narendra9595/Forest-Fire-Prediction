import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import lime
import lime.lime_tabular
from sklearn.metrics import accuracy_score, classification_report
import pickle

def train_explainable_model(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Calculate predictions on th etesing data set 
    y_pred = rf_model.predict(X_test)
    
    # Model performance
    print("Model Performance:")
    print(classification_report(y_test, y_pred))
    
    # SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    
    # LIME explainer
    featuer_names = list(X_train.columns)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['No Fire', 'Fire'],
        mode='classification',
        training_labels=y_train
    )
    
    return rf_model, X_test, y_test, shap_values, lime_explainer,featuer_names
