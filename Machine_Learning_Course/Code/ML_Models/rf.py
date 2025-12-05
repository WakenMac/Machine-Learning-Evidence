import time

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, roc_curve, auc, classification_report
)
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load a sample dataset
df = pd.read_csv('Machine_Learning_Course\Code\Dataset_Generation\main_dataset.csv')
X = df.iloc[:, 4:-3]  # Features
# y = df.iloc[:, -1] # Target variable (4 class)
y = df.iloc[:, -3] # Target variable (2 class)

# Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================================================================================
# For plotting

def heatmap(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    plot = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
    plt.title('0 = Hover | 1 = Press')
    return plot

# =============================================================================================

dt_model = DecisionTreeClassifier(random_state=77)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print(f"Random Forest Model Accuracy: {accuracy_score(y_test, dt_pred):.2f}")
heatmap(confusion_matrix(y_test, dt_pred))

# 3. Initialize the Random Forest Classifier
#    n_estimators: Number of trees in the forest (e.g., 100)
#    random_state: For reproducibility
rf_model = RandomForestClassifier(n_estimators=50, random_state=77)

# 4. Train the Random Forest model
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print(f"Random Forest Model Accuracy: {accuracy_score(y_test, rf_pred):.2f}")
print(classification_report(y_test, rf_pred))
heatmap(confusion_matrix(y_test, rf_pred))

# Training the SVM models
param_dist_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'poly', 'linear'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'degree': [2, 3, 4],
}

# Base SVM model with class_weight='balanced'
svm_base = SVC(probability=True, random_state=77, class_weight='balanced')

# RandomizedSearchCV for hyperparameter tuning
print("Running RandomizedSearchCV for SVM (20 iterations)...")
svm_search = RandomizedSearchCV(
    svm_base, param_dist_svm, n_iter=20, cv=3, 
scoring='f1_weighted', n_jobs=-1, random_state=77, verbose=1
)

start_time = time.time()
svm_search.fit(X_train, y_train)
train_time = time.time() - start_time

svm_sample = SVC(C=0.1, kernel='poly', gamma='scale', random_state=77)
svm_sample.fit(X_train, y_train)
svm_pred = svm_sample.predict(X_test)
print(f"SVM Model Accuracy: {accuracy_score(y_test, svm_pred):.2f}")
print(classification_report(y_test, svm_pred))
heatmap(confusion_matrix(y_test, svm_pred))

print(f"✅ SVM training complete ({train_time:.2f}s)")
print(f"   Best parameters: {svm_search.best_params_}")
print(f"   Best CV F1 score: {svm_search.best_score_:.4f}")

# Get best model
svm_model = svm_search.best_estimator_

# Example of making a prediction on new, unseen data
cols = ['tip2dip', 'tip2pip', 'tip2mcp', 'tip2wrist', 'disp', 'velocity_size', 'velocity_disp', 'accuracy_disp',
       'distance_cm']
new_data = [[]] # Example features for a new flower
prediction = rf_model.predict(new_data)
# cm = confusion_matrix(y_test, rf_y_pred)
# print(cm)
# print(classification_report(y_test, rf_y_pred))
print(f"Prediction for new data: {df.target_names[prediction[0]]}")
