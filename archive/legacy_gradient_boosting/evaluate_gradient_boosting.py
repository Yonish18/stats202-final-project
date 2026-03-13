from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "training.csv"


def add_feats(df):
    df = df.copy()
    for column in ["sig3", "sig4", "sig5", "sig6"]:
        if column in df.columns:
            df[f"log_{column}"] = np.log1p(df[column].astype(float))

    df["sig_ratio_21"] = df["sig2"] / (df["sig1"] + 1e-6)
    df["sig_sum_178"] = df["sig1"] + df["sig7"] + df["sig8"]
    df["hp_sig2"] = df["is_homepage"] * df["sig2"]
    return df


train = pd.read_csv(TRAIN_PATH)
y = train["relevance"].astype(int)

X = train.drop(columns=["relevance", "id"])
X = add_feats(X)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

gb = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    max_features=None,
    random_state=42,
)
gb.fit(X_train, y_train)

val_prob = gb.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0.3, 0.7, 81)
best_threshold = 0.5
best_accuracy = 0.0
for threshold in thresholds:
    predictions = (val_prob >= threshold).astype(int)
    accuracy = accuracy_score(y_val, predictions)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best validation accuracy: {best_accuracy:.4f} at threshold {best_threshold:.3f}")

val_pred = (val_prob >= best_threshold).astype(int)
print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_val, val_pred))
print("\nClassification report:")
print(classification_report(y_val, val_pred, digits=4))

importances = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 features:\n", importances.head(10))
