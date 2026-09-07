"""
Collaborative Filtering using matrix factorization (Truncated SVD).
Recommends items based on user-user and item-item rating patterns.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


class CollaborativeFilter:
    def __init__(self, n_components=20):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_item_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.users = None
        self.items = None

    def fit(self, ratings_df):
        """Train the model on a ratings DataFrame with user_id, item_id, rating columns."""
        # Build user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index="user_id", columns="item_id", values="rating"
        ).fillna(0)

        self.users = list(self.user_item_matrix.index)
        self.items = list(self.user_item_matrix.columns)

        # Matrix factorization
        self.user_factors = self.svd.fit_transform(self.user_item_matrix.values)
        self.item_factors = self.svd.components_

        # Reconstructed matrix
        self.predicted = np.dot(self.user_factors, self.item_factors)
        return self

    def recommend(self, user_id, top_k=5, exclude_seen=True):
        """Return top-k item recommendations for a given user."""
        if user_id not in self.users:
            # Cold-start: recommend globally popular items
            return self._cold_start(top_k)

        user_idx = self.users.index(user_id)
        scores = self.predicted[user_idx]

        # Exclude items the user has already rated
        if exclude_seen:
            seen = set(
                self.user_item_matrix.columns[
                    self.user_item_matrix.iloc[user_idx] > 0
                ]
            )
        else:
            seen = set()

        ranked = sorted(
            [(self.items[i], float(scores[i]))
             for i in range(len(self.items))
             if self.items[i] not in seen],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def _cold_start(self, top_k=5):
        """Recommend most popular items for new users."""
        popularity = self.user_item_matrix.sum(axis=0).sort_values(ascending=False)
        return [(item, float(score)) for item, score in popularity.head(top_k).items()]

    def evaluate(self, test_df):
        """Compute RMSE on a test set."""
        preds, actuals = [], []
        for _, row in test_df.iterrows():
            uid, iid, actual = row["user_id"], row["item_id"], row["rating"]
            if uid in self.users and iid in self.items:
                u_idx = self.users.index(uid)
                i_idx = self.items.index(iid)
                preds.append(self.predicted[u_idx, i_idx])
                actuals.append(actual)
        if not preds:
            return None
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        return rmse
