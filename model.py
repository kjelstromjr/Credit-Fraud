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

# Load datasets
print("Loading training data...")
train_df = pd.read_csv(Path(path) / "train.csv")

print("Loading testing data...")
test_df = pd.read_csv(Path(path) / "test.csv")

# Check missing values
print(f"Missing values: {train_df.isnull().sum().sum()}")

# Split features/labels
train_X = train_df.drop(columns=["is_fraud", "Unnamed: 0"])
train_y = train_df["is_fraud"]

test_X = test_df.drop(columns=["is_fraud", "Unnamed: 0"])
test_y = test_df["is_fraud"]

# ---------------- Feature Engineering ----------------

# Transaction datetime
for df in [train_X, test_X]:
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df.drop(columns=["trans_date_trans_time"], inplace=True)

# DOB → age
for df in [train_X, test_X]:
    df["dob"] = pd.to_datetime(df["dob"])
    df["age"] = (pd.Timestamp("today") - df["dob"]).dt.days // 365
    df.drop(columns=["dob"], inplace=True)

# Drop ID-like columns
drop_cols = ["cc_num", "first", "last", "trans_num", "street"]
train_X = train_X.drop(columns=drop_cols)
test_X = test_X.drop(columns=drop_cols)

# Target encoding
high_card_cols = ["city", "merchant", "job"]
target_encoder = TargetEncoder(cols=high_card_cols, smoothing=1.0)
train_X[high_card_cols] = target_encoder.fit_transform(train_X[high_card_cols], train_y)
test_X[high_card_cols] = target_encoder.transform(test_X[high_card_cols])

# One-hot encoding
low_card_cols = ["category", "gender", "state"]
train_X = pd.get_dummies(train_X, columns=low_card_cols)
test_X = pd.get_dummies(test_X, columns=low_card_cols)

# Align columns
train_X, test_X = train_X.align(test_X, join='left', axis=1, fill_value=0)

# ---------------- Scaling (preserve names) ----------------

feature_names = train_X.columns

scaler = StandardScaler()
train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=feature_names)
test_X = pd.DataFrame(scaler.transform(test_X), columns=feature_names)

# ---------------- Model Storage ----------------

model_scores = {}

# =========================================================
# Logistic Regression
# =========================================================

print("\nTraining Logistic Regression...")
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

# Feature importance (coefficients)
lr_coefs = pd.Series(lr.coef_[0], index=feature_names)
top_lr_feature = lr_coefs.abs().idxmax()
print(f"Most impactful feature (Logistic Regression): {top_lr_feature} (coef={lr_coefs[top_lr_feature]:.4f})")

# Optional top 5
print("Top 5 LR features:\n", lr_coefs.abs().sort_values(ascending=False).head(5))


# =========================================================
# Decision Tree
# =========================================================

print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(class_weight="balanced")
dt_model.fit(train_X, train_y)

print("Testing Decision Tree...")
dt_pred = dt_model.predict(test_X)
dt_scores = dt_model.predict_proba(test_X)[:, 1]
model_scores["Decision Tree"] = dt_scores

print("Accuracy:", accuracy_score(test_y, dt_pred))
print(confusion_matrix(test_y, dt_pred))
print(classification_report(test_y, dt_pred))
print("ROC-AUC:", roc_auc_score(test_y, dt_scores))

# Feature importance
dt_importances = pd.Series(dt_model.feature_importances_, index=feature_names)
top_dt_feature = dt_importances.idxmax()
print(f"Most impactful feature (Decision Tree): {top_dt_feature} (importance={dt_importances[top_dt_feature]:.4f})")

print("Top 5 DT features:\n", dt_importances.sort_values(ascending=False).head(5))


# =========================================================
# Tuned Decision Tree
# =========================================================

print("\nTraining Tuned Decision Tree...")
dt_tuned = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced"
)
dt_tuned.fit(train_X, train_y)

print("Testing Tuned Decision Tree...")
dt_tuned_pred = dt_tuned.predict(test_X)
dt_tuned_scores = dt_tuned.predict_proba(test_X)[:, 1]
model_scores["Tuned Decision Tree"] = dt_tuned_scores

print("Accuracy:", accuracy_score(test_y, dt_tuned_pred))
print(confusion_matrix(test_y, dt_tuned_pred))
print(classification_report(test_y, dt_tuned_pred))
print("ROC-AUC:", roc_auc_score(test_y, dt_tuned_scores))

# Feature importance
dt_tuned_importances = pd.Series(dt_tuned.feature_importances_, index=feature_names)
top_dt_tuned_feature = dt_tuned_importances.idxmax()
print(f"Most impactful feature (Tuned Decision Tree): {top_dt_tuned_feature} (importance={dt_tuned_importances[top_dt_tuned_feature]:.4f})")

print("Top 5 Tuned DT features:\n", dt_tuned_importances.sort_values(ascending=False).head(5))


# =========================================================
# SVM
# =========================================================

print("\nTraining SVM...")
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

# Feature importance (coefficients)
svm_coefs = pd.Series(svm.coef_[0], index=feature_names)
top_svm_feature = svm_coefs.abs().idxmax()
print(f"Most impactful feature (SVM): {top_svm_feature} (coef={svm_coefs[top_svm_feature]:.4f})")

print("Top 5 SVM features:\n", svm_coefs.abs().sort_values(ascending=False).head(5))


# =========================================================
# ROC Curve
# =========================================================

plt.figure(figsize=(7, 6))
for model_name, scores in model_scores.items():
    fpr, tpr, _ = roc_curve(test_y, scores)
    auc = roc_auc_score(test_y, scores)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=300)
plt.close()

print("Saved ROC curves to roc_curves.png")


# =========================================================
# Precision-Recall Curve
# =========================================================

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
plt.savefig("precision_recall_curves.png", dpi=300)
plt.close()

print("Saved Precision-Recall curves to precision_recall_curves.png")