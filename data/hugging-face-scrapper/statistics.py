import pandas as pd

# Load parquet
df = pd.read_parquet("hf_models.parquet")

# Basic cleanup
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# 1) Timeseries: models created per month
models_per_month = df.groupby(df['created_at'].dt.to_period('M')).size()
print("Models created per month:")
print(models_per_month)

# 2) Aggregation by provider: total downloads and likes
provider_stats = df.groupby('provider').agg({
    'downloads': 'sum',
    'likes': 'sum',
    'model_name': 'count'  # number of models
}).rename(columns={'model_name': 'model_count'})\
.sort_values('downloads', ascending=False)

print("\nProvider stats:")
print(len(provider_stats))

# 3) Task (pipeline_tag) popularity: average downloads & likes
task_stats = df.groupby('task').agg({
    'downloads': ['sum', 'median'],
    'likes': ['sum', 'median'],
    'model_name': 'count'
})
task_stats.columns = ['_'.join(col).strip() for col in task_stats.columns.values]
task_stats = task_stats.sort_values('downloads_sum', ascending=False)

print("\nTask popularity:")
print(task_stats)

