# Simple Heart Disease Prediction with Decision Tree

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# Load data
dt = pd.read_csv("Heart_disease_cleveland_new.csv")
print("Dataset shape:", dt.shape)

# Prepare features and target
X = dt.drop('target', axis=1)
y = dt['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Scatter plot of actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, alpha=0.6, label='Actual', color='blue')
plt.scatter(range(len(y_test_pred)), y_test_pred, alpha=0.6, label='Predicted', color='red', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('Heart Disease (0=No, 1=Yes)')
plt.title('Actual vs Predicted Heart Disease')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('scatter_plot.png')


# Decision tree visualization
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=['No Disease', 'Disease'], filled=True)
plt.savefig('decision_tree.png')

# R2 Score calculation
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Training R2 Score: {train_r2:.4f}")
print(f"Test R2 Score: {test_r2:.4f}")



