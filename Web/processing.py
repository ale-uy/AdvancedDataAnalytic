import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("HR_capstone_dataset.csv")

df.drop_duplicates(ignore_index=True, inplace=True)

df["work_accident"] = df["work_accident"].astype('object')
df["promotion_last_5years"] = df["promotion_last_5years"].astype('object')

df = pd.get_dummies(data=df, drop_first=True)

X = df.drop('left', axis=1)
y = df['left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

train = pd.concat([X_train, y_train], axis=1)

validate = pd.concat([X_test, y_test], axis=1)

X = train.drop(["left"], axis=1)
y = train["left"]

X_val = validate.drop(["left"], axis=1)
y_val = validate["left"]

model = xgb.XGBClassifier(objective='binary:logistic')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model.fit(X_train, y_train)

y_pred_val = model.predict(X_val)

print(classification_report(y_val, y_pred_val))

joblib.dump(model, 'model.joblib')