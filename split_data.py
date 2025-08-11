import pandas as pd
from sklearn.model_selection import train_test_split

# 1. load data
train = pd.read_csv("training.csv")

# 2. separate features and target
y = train["relevance"].astype(int)
X = train.drop(columns=["relevance", "id"])  # drop id because it's just an identifier

# 3. split into train & validation sets
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y,
    test_size=0.2,       # 20% for validation
    random_state=42,     # reproducible split
    stratify=y           # keep same proportion of 0/1 in both sets
)

# 4. print shapes
print("Training set:", X_tr.shape, y_tr.shape)
print("Validation set:", X_val.shape, y_val.shape)
