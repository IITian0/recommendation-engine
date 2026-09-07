"""
Hybrid Recommender — combines collaborative filtering and content-based filtering
using a weighted blend of normalised scores.
"""
import numpy as np


class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        """
        Args:
            cf_weight: weight for collaborative filtering (0-1)
            cb_weight: weight for content-based filtering (0-1)
        """
        assert abs(cf_weight + cb_weight - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.cf_model = None
        self.cb_model = None

    def fit(self, cf_model, cb_model):
        """Attach pre-trained sub-models."""
        self.cf_model = cf_model
        self.cb_model = cb_model
        return self

    def recommend(self, user_id, ratings_df, top_k=5):
        """Get hybrid recommendations for a user."""
        # Get recommendations from both models (fetch more to have overlap)
        cf_recs = self.cf_model.recommend(user_id, top_k=top_k * 3)
        cb_recs = self.cb_model.recommend(user_id, ratings_df, top_k=top_k * 3)

        # Build score dictionaries
        cf_scores = {iid: score for iid, score in cf_recs}
        cb_scores = {iid: score for iid, score in cb_recs}

        # Normalise scores to [0, 1]
        cf_vals = list(cf_scores.values())
        cb_vals = list(cb_scores.values())

        cf_max, cf_min = (max(cf_vals), min(cf_vals)) if cf_vals else (1, 0)
        cb_max, cb_min = (max(cb_vals), min(cb_vals)) if cb_vals else (1, 0)

        all_items = set(cf_scores.keys()) | set(cb_scores.keys())

        hybrid_scores = {}
        for iid in all_items:
            cf_norm = (cf_scores.get(iid, 0) - cf_min) / (cf_max - cf_min + 1e-10)
            cb_norm = (cb_scores.get(iid, 0) - cb_min) / (cb_max - cb_min + 1e-10)
            hybrid_scores[iid] = self.cf_weight * cf_norm + self.cb_weight * cb_norm

        ranked = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
