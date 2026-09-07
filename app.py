"""
Flask API for serving recommendations.
Endpoints:
  GET  /recommend/<user_id>     → hybrid recommendations
  GET  /recommend/cf/<user_id>  → collaborative filtering recommendations
  GET  /similar/<item_id>       → similar items (content-based)
  GET  /health                   → health check
"""
from flask import Flask, jsonify
from data import generate_sample_data
from collaborative_filter import CollaborativeFilter
from content_filter import ContentBasedFilter
from hybrid_recommender import HybridRecommender

app = Flask(__name__)

# ---- Load data & train models on startup ----
print("Generating sample data...")
users, items, ratings = generate_sample_data()

print("Training Collaborative Filtering model...")
cf_model = CollaborativeFilter(n_components=20)
cf_model.fit(ratings)

print("Training Content-Based Filtering model...")
cb_model = ContentBasedFilter()
cb_model.fit(items)

print("Building Hybrid Recommender...")
hybrid_model = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
hybrid_model.fit(cf_model, cb_model)

print("✅ All models loaded. API ready!\n")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models": ["collaborative", "content-based", "hybrid"]})


@app.route("/recommend/<user_id>", methods=["GET"])
def recommend(user_id):
    """Hybrid recommendations for a user."""
    recs = hybrid_model.recommend(user_id, ratings, top_k=5)
    result = [
        {"item_id": iid, "score": round(score, 4)}
        for iid, score in recs
    ]
    # Enrich with item metadata
    for r in result:
        row = items[items["item_id"] == r["item_id"]]
        if not row.empty:
            r["title"] = row.iloc[0]["title"]
            r["genre"] = row.iloc[0]["genre"]
    return jsonify({"user_id": user_id, "recommendations": result})


@app.route("/recommend/cf/<user_id>", methods=["GET"])
def recommend_cf(user_id):
    """Collaborative filtering recommendations."""
    recs = cf_model.recommend(user_id, top_k=5)
    result = [{"item_id": iid, "score": round(score, 4)} for iid, score in recs]
    return jsonify({"user_id": user_id, "recommendations": result})


@app.route("/similar/<item_id>", methods=["GET"])
def similar_items(item_id):
    """Find items similar to a given item (content-based)."""
    sim = cb_model.similar_items(item_id, top_k=5)
    result = [
        {"item_id": iid, "similarity": round(score, 4)}
        for iid, score in sim
    ]
    return jsonify({"item_id": item_id, "similar_items": result})


if __name__ == "__main__":
    import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
