import pandas as pd

# Read CSV (adjust path if needed)
df = pd.read_csv("hf_models.csv")

# Convert created_at to datetime if not already
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# Save as Parquet (compressed by default)
df.to_parquet("hf_models.parquet", index=False, compression='brotli')
