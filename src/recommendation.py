# rec_engine.py
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset

# Dummy supplement database with properties
supplements = pd.DataFrame({
    'supplement_id': [1, 2, 3],
    'name': ['Supplement A', 'Supplement B', 'Supplement C'],
    'benefit': ['Energy boost', 'Muscle growth', 'Digestive health']
})

# Dummy user-supplement interaction data
interactions_df = pd.DataFrame({
    'user_id': [1, 1, 2, 3],
    'supplement_id': [1, 2, 2, 3],
    'interaction': [1, 1, 1, 1]
})

# Initialize LightFM dataset and fit with dummy users and items.
dataset = Dataset()
# For this example, we pre-define user IDs (could be dynamic in a full implementation)
dataset.fit(
    users=[1, 2, 3],
    items=supplements['supplement_id']
)

# Build interactions matrix
(interactions_matrix, weights) = dataset.build_interactions(
    [(row['user_id'], row['supplement_id']) for _, row in interactions_df.iterrows()]
)

# Create and train a basic LightFM model
model = LightFM(loss='warp')
model.fit(interactions_matrix, epochs=10, num_threads=2)

def predict_for_user(user_id, num_recommendations=3):
    n_users, n_items = interactions_matrix.shape
    scores = model.predict(user_id, np.arange(n_items))
    top_items = np.argsort(-scores)[:num_recommendations]
    recommended_supplements = supplements.iloc[top_items]['name'].tolist()
    return recommended_supplements

def get_recommendations(user_profile):
    # For this initial version, simply map a new user to an existing user ID.
    # In production you would use user_profile to construct a new vector.
    # Here we randomly choose a user from our dummy dataset:
    dummy_user_id = 1
    recs = predict_for_user(dummy_user_id)
    return recs

if __name__ == "__main__":
    recs = get_recommendations({})
    print("Recommendations:", recs)
