from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "training.csv"


train = pd.read_csv(TRAIN_PATH)

y = train["relevance"].astype(int)
X = train.drop(columns=["relevance", "id"])

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=600,
    max_depth=None,
    min_samples_leaf=1,
    n_jobs=-1,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {accuracy:.4f}")

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_val, val_pred))

print("\nClassification report:")
print(classification_report(y_val, val_pred, digits=4))

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop features:\n", importances.head(10))
