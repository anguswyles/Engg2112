import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


np.random.seed(0)
n_samples = 1000

data = pd.DataFrame({
    "moisture": np.random.uniform(0.10, 0.95, n_samples),   
    "nitrogen": np.random.uniform(10, 120, n_samples),     
    "ph": np.random.uniform(5.0, 8.5, n_samples),           
    "temperature": np.random.uniform(15, 35, n_samples),    
    "rainfall": np.random.uniform(0, 200, n_samples)    
})


def assign_crop(row):
    if row["moisture"] > 0.70 and row["rainfall"] > 100:
        return "Rice"

    elif row["ph"] > 6.5 and 0.35 < row["moisture"] < 0.70:
        return "Wheat"

    elif row["temperature"] > 24 and row["nitrogen"] > 60:
        return "Corn"
    else:
        return "Soybean"

data["crop"] = data.apply(assign_crop, axis=1)


feature_columns = ["moisture", "nitrogen", "ph", "temperature", "rainfall"]
X = data[feature_columns]
y = data["crop"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)


model = RandomForestClassifier(
    n_estimators=100,
    random_state=0
)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nModel performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")

report = classification_report(y_test, y_pred, output_dict=True)

print("Performance per crop:")
for label in ["Corn", "Rice", "Soybean", "Wheat"]:
    if label in report:
        p = report[label]["precision"]
        r = report[label]["recall"]
        f1 = report[label]["f1-score"]
        print(f"  {label:<8} Precision: {p:.2f}  Recall: {r:.2f}  F1: {f1:.2f}")

P = report["macro avg"]["precision"]
R = report["macro avg"]["recall"]
F1 = report["macro avg"]["f1-score"]

print("\nOverall performance:")
print(f"  Precision: {P:.2f}")
print(f"  Recall:    {R:.2f}")
print(f"  F1:        {F1:.2f}")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature importance:")
feature_importance = pd.Series(
    model.feature_importances_,
    index=feature_columns
)
for feature, importance in feature_importance.items():
    print(f"{feature.title()}: {importance:.3f}")

example_inputs = pd.DataFrame([
    {
        "moisture": 0.82,
        "nitrogen": 75,
        "ph": 6.1,
        "temperature": 28,
        "rainfall": 150
    },
    {
        "moisture": 0.45,
        "nitrogen": 55,
        "ph": 7.2,
        "temperature": 20,
        "rainfall": 60
    },
    {
        "moisture": 0.30,
        "nitrogen": 85,
        "ph": 6.0,
        "temperature": 31,
        "rainfall": 40
    }
])

predictions = model.predict(example_inputs)

print("\nSample Predictions")
for i, pred in enumerate(predictions, start=1):
    print(f"Example {i}: Predicted crop = {pred}")

def recommend_crop(moisture, nitrogen, ph, temperature, rainfall):
    input_df = pd.DataFrame([{
        "moisture": moisture,
        "nitrogen": nitrogen,
        "ph": ph,
        "temperature": temperature,
        "rainfall": rainfall
    }])
    prediction = model.predict(input_df)[0]
    return prediction
