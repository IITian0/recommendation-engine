# Recommendation Engine

A complete hybrid recommendation engine built with **Collaborative Filtering** + **Content-Based Filtering**, including evaluation metrics and a Flask API.

## Project Structure

```
recommendation_engine/
├── data.py                    # Synthetic data generation (users, items, ratings)
├── collaborative_filter.py     # CF using Truncated SVD (matrix factorization)
├── content_filter.py           # CBF using TF-IDF + cosine similarity
├── hybrid_recommender.py       # Weighted blend of CF + CBF
├── evaluation.py               # Precision@k, Recall@k, RMSE
├── app.py                      # Flask API for serving recommendations
├── main.py                     # Demo script — train, evaluate, show recommendations
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the demo (trains models + evaluates + shows recommendations)
python main.py

# 3. Start the Flask API
python app.py
```

## API Endpoints

| Method | Endpoint                    | Description                          |
|--------|-----------------------------|--------------------------------------|
| GET    | `/health`                   | Health check                         |
| GET    | `/recommend/<user_id>`      | Hybrid recommendations (top 5)       |
| GET    | `/recommend/cf/<user_id>`   | Collaborative filtering (top 5)      |
| GET    | `/similar/<item_id>`        | Similar items (content-based)       |

### Example

```bash
curl http://localhost:5000/recommend/U1000
```

```json
{
  "user_id": "U1000",
  "recommendations": [
    {"item_id": "M142", "score": 0.87, "title": "Movie_42", "genre": "Action"},
    {"item_id": "M178", "score": 0.82, "title": "Movie_78", "genre": "Sci-Fi"},
    ...
  ]
}
```

## How It Works

### 1. Collaborative Filtering (SVD)
- Builds a user-item rating matrix
- Applies Truncated SVD to decompose into latent factors
- Reconstructs the matrix to predict missing ratings
- Recommends items with highest predicted scores

### 2. Content-Based Filtering (TF-IDF)
- Extracts features from item metadata (genre + year)
- Computes cosine similarity between items
- Recommends items similar to those the user has rated highly

### 3. Hybrid Recommender
- Combines CF and CBF scores with configurable weights (60/40 default)
- Normalises each model's scores to [0, 1] before blending
- Returns top-k items by hybrid score

### 4. Evaluation
- **Precision@k** — fraction of top-k recommendations that are relevant
- **Recall@k** — fraction of relevant items recovered in top-k
- **RMSE** — root mean squared error of predicted vs actual ratings

## Customisation

- **Use real data:** Replace `data.py` with your own dataset (CSV/DB). Keep columns: `user_id`, `item_id`, `rating` for ratings, and `item_id`, `title`, `genre` for items.
- **Adjust weights:** Change `cf_weight` / `cb_weight` in `HybridRecommender`.
- **Change model:** Swap SVD for ALS, NMF, or neural collaborative filtering.
- **Scale:** Add Redis for caching, PostgreSQL for data, Docker for deployment.
