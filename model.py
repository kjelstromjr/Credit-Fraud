import kagglehub
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from category_encoders import TargetEncoder
from pathlib import Path
import matplotlib.pyplot as plt

# Download latest version
path = kagglehub.dataset_download("kaushalnandania/credit-card-fraud-detection")

print("Path to dataset files:", path)

# Load in datasets for training and testing
print("Loading training data...")
train_df = pd.read_csv(Path(path) / "train.csv")

print("Loading testing data...")
test_df = pd.read_csv(Path(path) / "test.csv")

# Check for missing values
print(f"Missing values: {train_df.isnull().sum().sum()}")

# Data Cleaning
train_X = train_df.drop(columns=["is_fraud", "Unnamed: 0"])
train_y = train_df["is_fraud"]

test_X = test_df.drop(columns=["is_fraud", "Unnamed: 0"])
test_y = test_df["is_fraud"]

# Convert Date
for df in [train_X, test_X]:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df.drop(columns=["trans_date_trans_time"], inplace=True)

# Convert Date of Birth
for df in [train_X, test_X]:
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (pd.Timestamp("today") - df["dob"]).dt.days // 365
    df.drop(columns=["dob"], inplace=True)

# Drop ID columns (removed merchant and job so they can be encoded)
drop_cols = ["cc_num", "first", "last", "trans_num", "street"]
train_X = train_X.drop(columns=drop_cols)
test_X = test_X.drop(columns=drop_cols)

# Target encode high-cardinality columns
high_card_cols = ["city", "merchant", "job"]
target_encoder = TargetEncoder(cols=high_card_cols, smoothing=1.0)
train_X[high_card_cols] = target_encoder.fit_transform(train_X[high_card_cols], train_y)
test_X[high_card_cols] = target_encoder.transform(test_X[high_card_cols])

# One-hot encode low-cardinality columns
low_card_cols = ["category", "gender", "state"]
train_X = pd.get_dummies(train_X, columns=low_card_cols)
test_X = pd.get_dummies(test_X, columns=low_card_cols)

# Align columns in case train/test have different dummies (e.g. rare states)
train_X, test_X = train_X.align(test_X, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# Dictionary to collect scores for plotting
model_scores = {}

# ============== Logistic Regression ===============

print("Training Logistic Regression...")
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(train_X, train_y)

print("Testing Logistic Regression")
lr_pred = lr.predict(test_X)
lr_scores = lr.predict_proba(test_X)[:, 1]
model_scores["Logistic Regression"] = lr_scores

print("Accuracy:", accuracy_score(test_y, lr_pred))
print(confusion_matrix(test_y, lr_pred))
print(classification_report(test_y, lr_pred))
print("ROC-AUC:", roc_auc_score(test_y, lr_scores))

# ============== Decision Tree ===============

clf = DecisionTreeClassifier(class_weight="balanced")

print("Training Decision Tree...")
clf.fit(train_X, train_y)

print("Testing Decision Tree...")
dt_pred = clf.predict(test_X)
dt_scores = clf.predict_proba(test_X)[:, 1]
model_scores["Decision Tree"] = dt_scores

print("Accuracy:", accuracy_score(test_y, dt_pred))
print(confusion_matrix(test_y, dt_pred))
print(classification_report(test_y, dt_pred))
print("ROC-AUC:", roc_auc_score(test_y, dt_scores))

# =========== Tuned Decision Tree ============

clf = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced"
)

print("Training Tuned Decision Tree...")
clf.fit(train_X, train_y)

print("Testing Tuned Decision Tree...")
dt_tuned_pred = clf.predict(test_X)
dt_tuned_scores = clf.predict_proba(test_X)[:, 1]
model_scores["Tuned Decision Tree"] = dt_tuned_scores

print("Accuracy:", accuracy_score(test_y, dt_tuned_pred))
print(confusion_matrix(test_y, dt_tuned_pred))
print(classification_report(test_y, dt_tuned_pred))
print("ROC-AUC:", roc_auc_score(test_y, dt_tuned_scores))

# ============ SVM ================

print("Training SVM...")
svm = LinearSVC(class_weight='balanced')
svm.fit(train_X, train_y)

print("Testing SVM...")
svm_pred = svm.predict(test_X)
svm_scores = svm.decision_function(test_X)
model_scores["SVM"] = svm_scores

print("Accuracy:", accuracy_score(test_y, svm_pred))
print(confusion_matrix(test_y, svm_pred))
print(classification_report(test_y, svm_pred))
print("ROC-AUC:", roc_auc_score(test_y, svm_scores))

# ============ Combined ROC Curve Plot ================

plt.figure(figsize=(7, 6))
for model_name, scores in model_scores.items():
    fpr, tpr, _ = roc_curve(test_y, scores)
    auc = roc_auc_score(test_y, scores)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved ROC curves to roc_curves.png")

# ============ Combined Precision-Recall Curve Plot ================

plt.figure(figsize=(7, 6))
for model_name, scores in model_scores.items():
    precision, recall, _ = precision_recall_curve(test_y, scores)
    ap = average_precision_score(test_y, scores)
    plt.plot(recall, precision, label=f"{model_name} (AP = {ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("precision_recall_curves.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved Precision-Recall curves to precision_recall_curves.png")