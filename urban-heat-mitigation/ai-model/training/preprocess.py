import pandas as pd

# Load dataset
df = pd.read_csv("data/processed/final_dataset.csv")

# Remove missing values if any
df = df.dropna()

# Save cleaned dataset
df.to_csv(
    "data/processed/final_dataset_clean.csv",
    index=False
)

print("Preprocessing completed.")
print("Shape:", df.shape)