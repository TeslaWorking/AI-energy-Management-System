import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.tree import DecisionTreeRegressor

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv(r"C:\Users\user\Downloads\full combined dataset.csv")

# -----------------------------
# Drop unwanted columns
# -----------------------------

drop_columns = [
    "battery_id",
    "Timestamp",
    "fault_type"
]

for col in drop_columns:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# -----------------------------
# Target
# -----------------------------

target = "Electricity_Consumed"

X = df.drop(columns=[target])

y = df[target]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -----------------------------
# Decision Tree Regressor
# -----------------------------

model = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(y_test, prediction)

rmse = mse ** 0.5

r2 = r2_score(y_test, prediction)

accuracy = r2 * 100

print("\n==============================")

print("Decision Tree Results")

print("R2 Score :", round(r2,4))

print("Accuracy :", round(accuracy,2),"%")

print("MAE :", round(mae,4))

print("RMSE :", round(rmse,4))

print("==============================")

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "decision_tree_model.pkl")

print("Decision Tree Model Saved Successfully")
