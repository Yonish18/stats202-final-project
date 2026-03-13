from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "data" / "raw" / "training.csv"
TEST_PATH = ROOT / "data" / "raw" / "test.csv"
OUTPUT_PATH = ROOT / "results" / "submissions" / "logistic_regression_submission.csv"


train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

y = train["relevance"].astype(int)
X = train.drop(columns=["relevance", "id"])
X_test = test.drop(columns=["id"])

X_test = X_test[X.columns]

pipeline = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        (
            "logreg",
            LogisticRegressionCV(
                Cs=[0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                cv=5,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5000,
                scoring="accuracy",
                n_jobs=-1,
                refit=True,
            ),
        ),
    ]
)

pipeline.fit(X, y)

predictions = pipeline.predict(X_test).astype(int)

submission = pd.DataFrame({"id": test["id"], "relevance": predictions})
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)} with shape {submission.shape}")
