import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()))
sys.path.insert(0, str(pathlib.Path.cwd() / 'production_app'))

import mlflow, mlflow.sklearn
import pandas as pd

# Load features parquet
pq_path = pathlib.Path('data/features')
parquet_files = list(pq_path.glob('*.parquet'))
print("Parquet files:", parquet_files)

df = pd.read_parquet(parquet_files[0])
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("isRecommended value_counts:", df['isRecommended'].value_counts().to_dict())

# Load model
mlflow.set_tracking_uri("sqlite:///mlruns.db")
model = mlflow.sklearn.load_model("models:/best_optuna_model/latest")

cls1 = df[df['isRecommended'] == 1].drop(columns=['isRecommended'])
probs = model.predict_proba(cls1)[:, 1]
best_idx = probs.argmax()
best_row = cls1.iloc[best_idx]
print(f"\nBest prob: {probs[best_idx]:.4f}")
print("Best row values:")
for col, val in best_row.items():
    print(f"  {col}: {val}")
