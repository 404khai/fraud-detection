# Save as fraud_pipeline.py or run in a notebook cell-by-cell.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import seaborn as sns  # used only for nicer heatmap; matplotlib for other plots

# Optional: imbalanced-learn for SMOTE (install with: pip install imbalanced-learn)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    # print("imblearn not available. Run `pip install imbalanced-learn` to enable balancing options.")


# ----------------------
# 1) Load & Initial Inspection
# ----------------------
DATA_PATH = "transactions.csv"
assert os.path.exists(DATA_PATH), f"File not found: {DATA_PATH}"

df = pd.read_csv(DATA_PATH)
print("First 5 rows:")
display(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum().sort_values(ascending=False).head(30))

print("\nDescriptive stats for numeric columns:")
display(df.describe().T)


# ----------------------
# 2) Data Cleaning
# ----------------------
# Common ID columns to drop if present (you already dropped many)
cols_to_drop_candidates = ["TransactionID", "AccountID", "DeviceID", "MerchantID", "IP Address", "IP_Address", "ip"]
drop_cols = [c for c in cols_to_drop_candidates if c in df.columns]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")

# Remove exact duplicates
before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Dropped {before-after} duplicate rows")

# Fix date column: try to parse any date-like columns (common names)
date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
print("Candidate date columns:", date_cols)
for c in date_cols:
    try:
        df[c] = pd.to_datetime(df[c], errors='coerce')
    except Exception:
        pass

# If there is a 'TransactionDate' or 'Timestamp', ensure it's datetime
if "TransactionDate" in df.columns and not np.issubdtype(df["TransactionDate"].dtype, np.datetime64):
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")

# Target column: locate likely fraud column names
possible_targets = ["isFraud", "fraud", "is_fraud", "label", "Class", "Fraud"]
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break
assert target_col is not None, "No target column found. Modify possible_targets list to match your dataset's label column."

# Ensure target is binary 0/1
if df[target_col].dtype == object:
    # try converting 'Yes'/'No' or 'fraud' strings
    df[target_col] = df[target_col].map(lambda x: 1 if str(x).strip().lower() in ["1", "yes", "y", "true", "fraud", "fraudulent"] else 0)

df[target_col] = df[target_col].fillna(0).astype(int)
print("Target column:", target_col, "value counts:\n", df[target_col].value_counts())

# Handle missing values (policy: drop cols with >50% missing, otherwise impute)
missing_pct = df.isnull().mean()
cols_drop_high_missing = missing_pct[missing_pct > 0.5].index.tolist()
if cols_drop_high_missing:
    print("Dropping high-missing columns (>50% missing):", cols_drop_high_missing)
    df = df.drop(columns=cols_drop_high_missing)

# For remaining missing: numeric -> median, categorical -> 'missing'
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# remove target from num_cols if present
if target_col in num_cols:
    num_cols.remove(target_col)

print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# Impute numeric with median
for c in num_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna(df[c].median())

# Impute categorical with 'missing'
for c in cat_cols:
    if df[c].isnull().any():
        df[c] = df[c].fillna("missing")

# Confirm no remaining missing for core features (not always mandatory)
print("Missing after imputation (top 20):")
print(df.isnull().sum().sort_values(ascending=False).head(20))


# ----------------------
# 3) EDA (visualizations)
# ----------------------
# Basic target distribution
plt.figure(figsize=(5,4))
df[target_col].value_counts().plot(kind='bar')
plt.title("Target distribution (0 = legit, 1 = fraud)")
plt.xlabel("class")
plt.ylabel("count")
plt.show()

# Histograms for numeric features: show a few top numeric features
sample_num_cols = num_cols[:10]  # limit for plotting
df[sample_num_cols].hist(bins=30, figsize=(14,8))
plt.suptitle("Numeric feature distributions")
plt.show()

# Boxplot of transaction amount (if present)
if "TransactionAmount" in df.columns:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df, x=target_col, y="TransactionAmount")
    plt.title("Transaction amount vs Fraud")
    plt.show()

# Correlation matrix (numeric)
if len(num_cols) > 1:
    corr = df[num_cols + [target_col]].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation heatmap (numeric features + target)")
    plt.show()

# Fraud vs non-fraud comparison for the amount (violin)
if "TransactionAmount" in df.columns:
    plt.figure(figsize=(8,4))
    sns.violinplot(x=target_col, y="TransactionAmount", data=df, scale="width")
    plt.title("TransactionAmount distribution by class")
    plt.show()

# Identify outliers with IQR for TransactionAmount (if exists)
if "TransactionAmount" in df.columns:
    q1 = df["TransactionAmount"].quantile(0.25)
    q3 = df["TransactionAmount"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df["TransactionAmount"] < lower) | (df["TransactionAmount"] > upper)]
    print(f"TransactionAmount outliers: {len(outliers)} rows ({len(outliers)/len(df):.2%})")


# ----------------------
# 4) Feature Engineering & Wrangling
# ----------------------
# Example features: hour of day, day of week, is_weekend, log amount, amount buckets
if "TransactionDate" in df.columns and np.issubdtype(df["TransactionDate"].dtype, np.datetime64):
    df["hour"] = df["TransactionDate"].dt.hour
    df["dayofweek"] = df["TransactionDate"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5,6]).astype(int)
else:
    print("No TransactionDate datetime column found or not parsed. Skipping time-based features.")

if "TransactionAmount" in df.columns:
    # log transform (add small epsilon to avoid log(0))
    eps = 1e-9
    df["log_amount"] = np.log1p(df["TransactionAmount"].clip(lower=0) + eps)
    # bucket amount into quantiles
    df["amt_qcut"] = pd.qcut(df["TransactionAmount"].rank(method="first"), q=5, labels=[f"Q{i}" for i in range(1,6)])
else:
    print("No TransactionAmount column, skipping amount features")

# Example ratio: if there is an 'AccountBalance' or 'AvgDailyBalance' column
if "AccountBalance" in df.columns and "TransactionAmount" in df.columns:
    df["amount_to_balance_ratio"] = df["TransactionAmount"] / (df["AccountBalance"].replace({0:np.nan}) + eps)
    df["amount_to_balance_ratio"] = df["amount_to_balance_ratio"].fillna(0)

# One-hot or ordinal encode small-cardinality categorical variables
# Gather final feature list automatically
exclude = [target_col]
final_features = [c for c in df.columns if c not in exclude and not c in date_cols]
print("Candidate features for modeling (first 40):", final_features[:40])


# ----------------------
# 5) Baseline Classification Model
# ----------------------
# Choose features & automatic handling for categorical vs numeric
X = df[final_features].copy()
y = df[target_col].values

# Convert any remaining object dtypes to categorical
for c in X.select_dtypes(include=['object']).columns:
    # if too many unique values, drop or reduce cardinality
    if X[c].nunique() > 30:
        print(f"Column {c} has high cardinality ({X[c].nunique()} unique). Dropping/aggregating may be necessary.")
        # simple fallback: drop for baseline
        X = X.drop(columns=[c])
    else:
        X[c] = X[c].astype('category')

# Recompute types
num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(include=['category', 'object']).columns.tolist()
print("Final numeric features:", num_features)
print("Final categorical features:", cat_features)

# Train/test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train class distribution:", np.bincount(y_train))
print("Test class distribution:", np.bincount(y_test))

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ],
    remainder='drop'
)

# Baseline models: Logistic Regression and Random Forest in pipelines
log_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])

rf_pipe = Pipeline(steps=[
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
])

# Fit Logistic Regression baseline
log_pipe.fit(X_train, y_train)
y_pred_log = log_pipe.predict(X_test)
y_prob_log = log_pipe.predict_proba(X_test)[:,1]

# Fit Random Forest baseline
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)
y_prob_rf = rf_pipe.predict_proba(X_test)[:,1]

# Evaluation helper
def evaluate_model(y_true, y_pred, y_prob, model_name="model"):
    print(f"=== Evaluation: {model_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1:", f1_score(y_true, y_pred, zero_division=0))
    print("ROC AUC:", roc_auc_score(y_true, y_prob))
    print("\nClassification report:\n", classification_report(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({model_name})")
    plt.legend()
    plt.show()

# Evaluate
evaluate_model(y_test, y_pred_log, y_prob_log, "LogisticRegression (baseline)")
evaluate_model(y_test, y_pred_rf, y_pred_rf, y_prob_rf, "RandomForest (baseline)")

# Feature importance for RandomForest (after preprocessing we need a trick)
# Extract feature names from preprocessor
def get_feature_names_from_preprocessor(preprocessor):
    feature_names = []
    # numeric
    if preprocessor.transformers_[0][2]:
        feature_names.extend(preprocessor.transformers_[0][2])
    # categorical onehot feature names
    cat_transformer = preprocessor.transformers_[1][1]
    cat_cols = preprocessor.transformers_[1][2]
    try:
        ohe = cat_transformer.named_steps['onehot']
        ohe_feature_names = ohe.get_feature_names_out(cat_cols)
        feature_names.extend(ohe_feature_names)
    except Exception as e:
        # fallback: use raw cat_cols
        feature_names.extend(cat_cols)
    return feature_names

try:
    feat_names = get_feature_names_from_preprocessor(preprocessor)
    rf = rf_pipe.named_steps['clf']
    importances = rf.feature_importances_
    fi = pd.DataFrame({'feature': feat_names, 'importance': importances})
    fi = fi.sort_values('importance', ascending=False).head(30)
    print("Top features by RandomForest importance:")
    display(fi)
    plt.figure(figsize=(8,6))
    sns.barplot(data=fi, x='importance', y='feature')
    plt.title("Top feature importances (RandomForest)")
    plt.show()
except Exception as e:
    print("Could not extract feature importances due to:", e)


# ----------------------
# 6) Interpretation & Error analysis
# ----------------------
# Find false negatives (actual=1 but predicted=0) and false positives
def error_analysis(X_test_df, y_test, y_pred, X_full=df):
    idxs = np.where((y_test == 1) & (y_pred == 0))[0]
    print("False negatives:", len(idxs))
    if len(idxs) > 0:
        # show a few rows to inspect
        display(X_test_df.iloc[idxs[:10]])
    idxs_fp = np.where((y_test == 0) & (y_pred == 1))[0]
    print("False positives:", len(idxs_fp))
    if len(idxs_fp) > 0:
        display(X_test_df.iloc[idxs_fp[:10]])

# For error analysis we need X_test as a dataframe (we have it)
error_analysis(X_test.reset_index(drop=True), y_test, y_pred_rf)


# ----------------------
# 7) Optional Extensions (cross-validation, balancing, threshold tuning)
# ----------------------
# Cross-validation example (stratified) for RandomForest
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring='roc_auc')
print("RandomForest CV ROC AUC scores:", cv_scores)
print("Mean ROC AUC:", cv_scores.mean())

# Balancing with SMOTE example (if imblearn available)
if IMBLEARN_AVAILABLE:
    print("imblearn available: showing SMOTE pipeline example")
    smote_pipe = ImbPipeline(steps=[
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    smote_pipe.fit(X_train, y_train)
    y_pred_smote = smote_pipe.predict(X_test)
    y_prob_smote = smote_pipe.predict_proba(X_test)[:,1]
    evaluate_model(y_test, y_pred_smote, y_prob_smote, "RF + SMOTE")
else:
    print("imblearn not installed. To try SMOTE: pip install imbalanced-learn")

# Threshold tuning example for precision/recall tradeoff (using model probabilities)
probs = y_prob_rf
fpr, tpr, thresholds = roc_curve(y_test, probs)
# choose threshold that gives ~90% recall if possible
for thresh in [0.5, 0.4, 0.3, 0.2, 0.1]:
    y_pred_thresh = (probs >= thresh).astype(int)
    print(f"Threshold {thresh}: Precision={precision_score(y_test, y_pred_thresh, zero_division=0):.3f}, Recall={recall_score(y_test, y_pred_thresh, zero_division=0):.3f}, F1={f1_score(y_test, y_pred_thresh, zero_division=0):.3f}")


# Optionally save the best model
import joblib
joblib.dump(rf_pipe, "rf_baseline_pipeline.joblib")
print("Saved RandomForest pipeline to rf_baseline_pipeline.joblib")
