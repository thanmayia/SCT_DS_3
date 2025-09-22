

# decision_tree_bank.py
# Task 03: Decision Tree Classifier on Bank Marketing Dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt
import os
import requests
import zipfile

# ---------------------------
# Step 1: Download dataset if not present
# ---------------------------
dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
zip_file = "bank.zip"
csv_file = "bank.csv"

if not os.path.exists(csv_file):
    print("Downloading dataset...")
    r = requests.get(dataset_url)
    with open(zip_file, "wb") as f:
        f.write(r.content)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    os.remove(zip_file)

# ---------------------------
# Step 2: Load dataset
# ---------------------------
df = pd.read_csv(csv_file, delimiter=";")
print("\nFirst 5 rows of dataset:")
print(df.head())

# ---------------------------
# Step 3: Encode categorical data
# ---------------------------
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

print("\nDataset after encoding:")
print(df.head())

# ---------------------------
# Step 4: Define features and target
# ---------------------------
X = df.drop("y", axis=1)
y = df["y"]

# ---------------------------
# Step 5: Split dataset
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 6: Train Decision Tree
# ---------------------------
model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Step 7: Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# Step 8: Evaluation
# ---------------------------
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# Step 9: Plot Decision Tree
# ---------------------------
plt.figure(figsize=(20, 10))
tree.plot_tree(model, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.savefig("decision_tree.png")
print("\nDecision tree visualization saved as 'decision_tree.png'")
plt.show()
