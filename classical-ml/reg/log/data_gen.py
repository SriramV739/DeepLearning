import pandas as pd
from sklearn.datasets import make_classification

# Generate synthetic binary-classification data
X, y = make_classification(
    n_samples=5000,          # Number of rows
    n_features=6,            # Six feature columns
    n_informative=4,         # Four features strongly affect the label
    n_redundant=1,           # One feature is correlated with other features
    n_repeated=0,
    n_classes=2,             # Labels are 0 or 1
    weights=[0.5, 0.5],      # Balanced classes
    class_sep=1.2,           # Separation between the two classes
    flip_y=0.03,             # Add 3% label noise for realism
    random_state=42
)

# Create a DataFrame
df = pd.DataFrame(
    X,
    columns=[
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6"
    ]
)

df["label"] = y

# Shuffle the rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the dataset
df.to_csv("logistic_regression_data.csv", index=False)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts())