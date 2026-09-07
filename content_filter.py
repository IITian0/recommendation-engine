"""
Content-Based Filtering using TF-IDF on item features (genre, year).
Recommends items similar to those a user has liked in the past.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    def __init__(self):
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.item_features = None
        self.similarity_matrix = None
        self.items = None
        self.item_ids = None

    def fit(self, items_df):
        """Build TF-IDF features and similarity matrix from item metadata."""
        # Combine genre and year into a single text feature
        items_df = items_df.copy()
        items_df["features"] = items_df["genre"] + " " + items_df["year"].astype(str)

        self.items = items_df
        self.item_ids = list(items_df["item_id"])

        tfidf_matrix = self.tfidf.fit_transform(items_df["features"])
        self.item_features = tfidf_matrix
        self.similarity_matrix = cosine_similarity(tfidf_matrix)
        return self

    def recommend(self, user_id, ratings_df, top_k=5):
        """Recommend items based on content similarity to user's highly-rated items."""
        user_ratings = ratings_df[ratings_df["user_id"] == user_id]

        if user_ratings.empty:
            # Cold-start: recommend top-rated items overall
            avg_ratings = ratings_df.groupby("item_id")["rating"].mean()
            return [(iid, float(score)) for iid, score in avg_ratings.nlargest(top_k).items()]

        # Weighted average of similarities to items the user liked (rating >= 3.5)
        liked = user_ratings[user_ratings["rating"] >= 3.5]
        if liked.empty:
            liked = user_ratings.sort_values("rating", ascending=False).head(3)

        scores = np.zeros(len(self.item_ids))
        seen = set(user_ratings["item_id"].values)

        for _, row in liked.iterrows():
            iid = row["item_id"]
            if iid in self.item_ids:
                idx = self.item_ids.index(iid)
                weight = row["rating"]
                scores += weight * self.similarity_matrix[idx]

        # Normalise
        scores = scores / (len(liked) + 1e-10)

        ranked = sorted(
            [(self.item_ids[i], float(scores[i]))
             for i in range(len(self.item_ids))
             if self.item_ids[i] not in seen],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]

    def similar_items(self, item_id, top_k=5):
        """Find items most similar to a given item."""
        if item_id not in self.item_ids:
            return []
        idx = self.item_ids.index(item_id)
        sim_scores = self.similarity_matrix[idx]
        ranked = sorted(
            [(self.item_ids[i], float(sim_scores[i]))
             for i in range(len(self.item_ids)) if i != idx],
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
