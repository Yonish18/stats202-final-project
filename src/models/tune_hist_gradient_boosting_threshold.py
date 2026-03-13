from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.utils.feature_engineering import add_engineered_features


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "training.csv"


train = pd.read_csv(TRAIN_PATH)
y = train["relevance"].astype(int)
X = add_engineered_features(
    train.drop(columns=["relevance"]),
    include_query_context=True,
    include_query_aggregates=False,
)

groups = X["query_id"].values
X = X.drop(columns=["id", "url_id", "query_id"], errors="ignore")

hgb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.06,
    max_iter=350,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
)

gkf = GroupKFold(n_splits=3)
probs = []
targets = []

for train_idx, val_idx in gkf.split(X, y, groups):
    hgb.fit(X.iloc[train_idx], y.iloc[train_idx])
    fold_probs = hgb.predict_proba(X.iloc[val_idx])[:, 1]
    probs.append(fold_probs)
    targets.append(y.iloc[val_idx].values)

probs = np.concatenate(probs)
targets = np.concatenate(targets)

best_threshold = 0.5
best_accuracy = 0.0
for threshold in np.linspace(0.45, 0.55, 41):
    accuracy = ((probs >= threshold).astype(int) == targets).mean()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold

print(f"Best CV accuracy: {best_accuracy:.4f} at threshold {best_threshold:.3f}")
