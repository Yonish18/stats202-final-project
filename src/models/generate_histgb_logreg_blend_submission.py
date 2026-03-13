from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.feature_engineering import add_engineered_features


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "training.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
OUTPUT_DIR = ROOT / "results" / "submissions"


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

y = train["relevance"].astype(int)
X_features = add_engineered_features(
    train.drop(columns=["relevance"]),
    include_query_context=True,
    include_query_aggregates=True,
)
test_features = add_engineered_features(
    test.copy(),
    include_query_context=True,
    include_query_aggregates=True,
)

drop_columns = ["id", "url_id", "query_id"]
X = X_features.drop(columns=[column for column in drop_columns if column in X_features.columns])
X_test = test_features.drop(
    columns=[column for column in drop_columns if column in test_features.columns],
    errors="ignore",
)
X_test = X_test[X.columns]

hgb = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.06,
    max_iter=350,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42,
)

logit = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=5000,
                solver="lbfgs",
                n_jobs=-1,
            ),
        ),
    ]
)

hgb.fit(X, y)
logit.fit(X, y)

hgb_probs = hgb.predict_proba(X_test)[:, 1]
logit_probs = logit.predict_proba(X_test)[:, 1]
base_predictions = (hgb_probs >= 0.5).astype(int)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

weights = [0.90, 0.92, 0.94, 0.96, 0.98]
thresholds = [0.498, 0.499, 0.500, 0.501, 0.502]

created_files = []
for weight in weights:
    blended_probs = weight * hgb_probs + (1.0 - weight) * logit_probs
    for threshold in thresholds:
        predictions = (blended_probs >= threshold).astype(int)
        flips = int(np.sum(predictions != base_predictions))
        submission = pd.DataFrame({"id": test["id"], "relevance": predictions})
        output_path = OUTPUT_DIR / (
            f"histgb_logreg_blend_w{weight:.2f}_thr{threshold:.3f}_flips{flips}.csv"
        )
        submission.to_csv(output_path, index=False)
        created_files.append((output_path, flips))
        print(
            f"Wrote {output_path.relative_to(ROOT)} "
            f"(changed {flips} rows vs HistGradientBoosting @ 0.500)"
        )

baseline_submission = pd.DataFrame({"id": test["id"], "relevance": base_predictions})
baseline_path = OUTPUT_DIR / "histgb_base_thr0.500.csv"
baseline_submission.to_csv(baseline_path, index=False)
print(f"Wrote {baseline_path.relative_to(ROOT)}")

recommended = sorted(created_files, key=lambda item: (item[1], item[0].name))
print("\nSuggested files to review first:")
for output_path, flips in recommended[:6]:
    print(f"  {output_path.name} (flips={flips})")
