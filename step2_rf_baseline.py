# step2_rf_baseline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# 1) load data
train = pd.read_csv("training.csv")

# 2) features/target
y = train["relevance"].astype(int)
X = train.drop(columns=["relevance", "id"])  # drop 'id' (just an identifier)

# 3) split (you already did this, but keeping it self-contained)
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4) model (book-friendly baseline)
model = RandomForestClassifier(
    n_estimators=600,      # number of trees
    max_depth=None,        # let trees grow fully (can tune later)
    min_samples_leaf=1,    # can increase to reduce overfitting
    n_jobs=-1,             # use all CPU cores
    class_weight="balanced",  # handle slight class imbalance
    random_state=42
)

# 5) train
model.fit(X_tr, y_tr)

# 6) evaluate
val_pred = model.predict(X_val)
acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {acc:.4f}")

print("\nConfusion matrix (rows=true, cols=pred):")
print(confusion_matrix(y_val, val_pred))

print("\nClassification report:")
print(classification_report(y_val, val_pred, digits=4))

# 7) quick feature importance peek
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop features:\n", importances.head(10))
