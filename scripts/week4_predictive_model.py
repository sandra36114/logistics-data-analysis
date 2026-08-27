import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

file_path = "data/smart_logistics_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)

# --------------------------------------------------
# 2. Data Preparation
# --------------------------------------------------

# Convert timestamp
df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Extract useful time features
df["Month"] = df["Timestamp"].dt.month
df["Hour"] = df["Timestamp"].dt.hour

# Fill missing delay reasons
df["Logistics_Delay_Reason"] = (
    df["Logistics_Delay_Reason"]
    .fillna("Unknown")
)

# --------------------------------------------------
# 3. Select Features
# --------------------------------------------------

features = [
    "Inventory_Level",
    "Temperature",
    "Humidity",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Asset_Utilization",
    "Demand_Forecast",
    "Month",
    "Hour"
]

X = df[features]

y = df["Logistics_Delay"]

print("\nFeatures:")
print(features)

print("\nTarget Distribution:")
print(y.value_counts())

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 5. Feature Scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 6. Logistic Regression Model
# --------------------------------------------------

logistic_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test_scaled
)

logistic_probabilities = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]

# --------------------------------------------------
# 7. Random Forest Model
# --------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)

rf_probabilities = rf_model.predict_proba(
    X_test
)[:, 1]

# --------------------------------------------------
# 8. Evaluation Function
# --------------------------------------------------

def evaluate_model(
    model_name,
    y_true,
    predictions,
    probabilities
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    print("\n================================")
    print(model_name)
    print("================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            zero_division=0
        )
    )

    return [
        model_name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]


# --------------------------------------------------
# 9. Evaluate Both Models
# --------------------------------------------------

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_predictions,
    logistic_probabilities
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_predictions,
    rf_probabilities
)

# --------------------------------------------------
# 10. Model Comparison
# --------------------------------------------------

comparison = pd.DataFrame(
    [
        logistic_results,
        rf_results
    ],
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC"
    ]
)

print("\n========== MODEL COMPARISON ==========")
print(comparison)

# Create output folder
import os

os.makedirs(
    "model_outputs",
    exist_ok=True
)

comparison.to_csv(
    "model_outputs/model_comparison.csv",
    index=False
)

# --------------------------------------------------
# 11. Cross Validation
# --------------------------------------------------

print("\n========== CROSS VALIDATION ==========")

rf_cv_scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=5,
    scoring="f1"
)

print("Random Forest F1 CV Scores:")
print(rf_cv_scores)

print(
    "Mean CV F1 Score:",
    round(rf_cv_scores.mean(), 4)
)

# --------------------------------------------------
# 12. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    rf_predictions
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.colorbar()

plt.xticks(
    [0, 1],
    ["No Delay", "Delay"]
)

plt.yticks(
    [0, 1],
    ["No Delay", "Delay"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "model_outputs/confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 13. Feature Importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

feature_importance.to_csv(
    "model_outputs/feature_importance.csv",
    index=False
)

plt.figure(figsize=(9, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "model_outputs/feature_importance.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 14. Final Model Selection
# --------------------------------------------------

best_model = comparison.sort_values(
    by="F1_Score",
    ascending=False
).iloc[0]

print("\n========== BEST MODEL ==========")

print(
    "Selected Model:",
    best_model["Model"]
)

print(
    "F1 Score:",
    round(best_model["F1_Score"], 4)
)

print(
    "ROC-AUC:",
    round(best_model["ROC_AUC"], 4)
)

print("\n========== WEEK 4 ANALYSIS COMPLETED ==========")

print("Outputs saved in:")
print("model_outputs/")