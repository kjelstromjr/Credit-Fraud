import kagglehub
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# Download latest version
path = kagglehub.dataset_download("kaushalnandania/credit-card-fraud-detection")

print("Path to dataset files:", path)

# Load in datasets for training and testing
print("Loading training data...")
train_df = pd.read_csv(path + '\\train.csv')

print("Loading testing data...")
test_df = pd.read_csv(path + '\\test.csv')

# Check for missing values
print(f"Missing values: {train_df.isnull().sum().sum()}")

# Data Cleaning
train_X = train_df.drop(columns=["is_fraud", "Unnamed: 0"])
train_y = train_df["is_fraud"]

test_X = test_df.drop(columns=["is_fraud", "Unnamed: 0"])
test_y = test_df["is_fraud"]

# Convert Date
train_X["trans_date_trans_time"] = pd.to_datetime(train_X["trans_date_trans_time"])

train_X["hour"] = train_X["trans_date_trans_time"].dt.hour
train_X["day"] = train_X["trans_date_trans_time"].dt.day
train_X["month"] = train_X["trans_date_trans_time"].dt.month
train_X["dayofweek"] = train_X["trans_date_trans_time"].dt.dayofweek

train_X = train_X.drop(columns=["trans_date_trans_time"])

test_X["trans_date_trans_time"] = pd.to_datetime(test_X["trans_date_trans_time"])

test_X["hour"] = test_X["trans_date_trans_time"].dt.hour
test_X["day"] = test_X["trans_date_trans_time"].dt.day
test_X["month"] = test_X["trans_date_trans_time"].dt.month
test_X["dayofweek"] = test_X["trans_date_trans_time"].dt.dayofweek

test_X = test_X.drop(columns=["trans_date_trans_time"])

# Convert Date of Birth
train_X["dob"] = pd.to_datetime(train_X["dob"])

train_X["age"] = (pd.Timestamp("today") - train_X["dob"]).dt.days // 365
train_X = train_X.drop(columns=["dob"])

test_X["dob"] = pd.to_datetime(test_X["dob"])

test_X["age"] = (pd.Timestamp("today") - test_X["dob"]).dt.days // 365
test_X = test_X.drop(columns=["dob"])

# Convert Strings
train_X = pd.get_dummies(train_X, columns=[
    # "merchant",
    "category",
    "gender",
    "city",
    "state"
    # "job"
])

test_X = pd.get_dummies(test_X, columns=[
    # "merchant",
    "category",
    "gender",
    "city",
    "state"
    # "job"
])

# Drop ID columns
train_X = train_X.drop(columns=[
    "cc_num",
    "first",
    "last",
    "trans_num",
    "street",
    "merchant",
    "job"
    # "city"
])

test_X = test_X.drop(columns=[
    "cc_num",
    "first",
    "last",
    "trans_num",
    "street",
    "merchant",
    "job"
    # "city"
])

train_X, test_X = train_X.align(test_X, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)

# ============== Logistic Regression ===============

print("Training Logistic Regression...")
lr = LogisticRegression(class_weight="balanced", max_iter=1000)
lr.fit(train_X, train_y)

print("Testing Logistic Regression")
lr_pred = lr.predict(test_X)
lr_scores = lr.predict_proba(test_X)[:, 1]

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
dt_scores = clf.predict_proba(test_X)[:, 1]  # probability of positive class

print("Accuracy:", accuracy_score(test_y, dt_pred))
print(confusion_matrix(test_y, dt_pred))
print(classification_report(test_y, dt_pred))
print("ROC-AUC:", roc_auc_score(test_y, dt_scores))  # use probabilities

# =========== Tuned Decision Tree ============

clf = DecisionTreeClassifier(
    max_depth=8,
    min_samples_leaf=20,
    class_weight="balanced"
)

print("Training Tuned Decision Tree...")
clf.fit(train_X, train_y)

print("Testing Tuned Decision Tree...")
dt_pred = clf.predict(test_X)
dt_scores = clf.predict_proba(test_X)[:, 1]  # probability of positive class

print("Accuracy:", accuracy_score(test_y, dt_pred))
print(confusion_matrix(test_y, dt_pred))
print(classification_report(test_y, dt_pred))
print("ROC-AUC:", roc_auc_score(test_y, dt_scores))  # use probabilities

# ============ SVM ================

print("Training SVM...")
model = LinearSVC(class_weight='balanced')
model.fit(train_X, train_y)

print("Testing SVM...")
y_pred = model.predict(test_X)
y_scores = model.decision_function(test_X)  # continuous scores for ROC-AUC

print("Accuracy:", accuracy_score(test_y, y_pred))
print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("ROC-AUC:", roc_auc_score(test_y, y_scores))  # use scores, not y_pred