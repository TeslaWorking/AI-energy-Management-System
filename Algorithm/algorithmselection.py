import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
df = pd.read_csv("/content/full combined dataset.csv")
drop_columns = [
    "battery_id",
    "Timestamp",
    "fault_type"
]
for col in drop_columns:
    if col in df.columns:
        df.drop(columns=col, inplace=True)
target = "Electricity_Consumed"
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
model = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42

)
model.fit(X_train, y_train)
prediction = model.predict(X_test)
