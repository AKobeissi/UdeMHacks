# rec_engine.py
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import implicit

# Dummy supplement database with properties
supplements = pd.DataFrame({
    'supplement_id': [1, 2, 3],
    'name': ['Supplement A', 'Supplement B', 'Supplement C'],
    'benefit': ['Energy boost', 'Muscle growth', 'Digestive health']
})

# Create a dummy user-supplement interactions matrix.
# Note: Implicit expects an item-user matrix.
# We create a 3x3 matrix (3 supplements x 3 users) as an example.
data = np.array([
    [1, 0, 1],  # Supplement A interactions with user0, user1, user2
    [0, 1, 0],  # Supplement B interactions
    [1, 1, 0]   # Supplement C interactions
])
sparse_data = sparse.csr_matrix(data)

# Train an Alternating Least Squares (ALS) model using Implicit
model = implicit.als.AlternatingLeastSquares(factors=10, regularization=0.1, iterations=20)
model.fit(sparse_data)

def predict_for_user(user_id, num_recommendations=3):
    """
    Given a user_id (which corresponds to a column in our item-user matrix),
    predict top supplements using the ALS model.
    """
    # Extract the user's interaction vector from the transpose (user-item matrix)
    user_interactions = sparse_data.T.tocsr()[user_id]
    
    # Generate recommendations.
    # Note: filter_already_liked_items is set to False for this demo.
    recs = model.recommend(user_id, user_interactions, N=num_recommendations, filter_already_liked_items=False)
    
    # recs is a list of (item_index, score) pairs.
    recommended_names = []
    for item_idx, score in recs:
        recommended_names.append(supplements.iloc[item_idx]['name'])
    return recommended_names

def get_recommendations(user_profile):
    # For this demo, we simply choose a dummy user id.
    # In a production scenario, you would convert the user_profile into an interaction vector.
    dummy_user_id = 0  # Using the first user in our dummy data.
    recs = predict_for_user(dummy_user_id)
    return recs

if __name__ == "__main__":
    recs = get_recommendations({})
    print("Recommendations:", recs)
