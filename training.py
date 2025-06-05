# training.py
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("workout_df.csv")[["disease", "workout"]]

# Pivot to user-item matrix (disease vs workout)
pivot = pd.crosstab(df['disease'], df['workout'])

# Compute cosine similarity between diseases
similarity = cosine_similarity(pivot)

# Save similarity matrix and other required lists
joblib.dump(similarity, "similarity_matrix.pkl")
joblib.dump(pivot.index.tolist(), "disease_list.pkl")
joblib.dump(pivot.columns.tolist(), "workout_list.pkl")

print("Training completed and files saved!")
