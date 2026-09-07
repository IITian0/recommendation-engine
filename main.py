"""
Main script — trains all models, evaluates them, and shows sample recommendations.
Run:  python main.py
"""
from data import generate_sample_data
from collaborative_filter import CollaborativeFilter
from content_filter import ContentBasedFilter
from hybrid_recommender import HybridRecommender
from evaluation import evaluate_model, train_test_split

def main():
    # 1. Generate data
    print("=" * 60)
    print("  RECOMMENDATION ENGINE — DEMO")
    print("=" * 60)

    users, items, ratings = generate_sample_data()
    print(f"\n📊 Data: {len(users)} users, {len(items)} items, {len(ratings)} ratings")
    print(f"   Rating range: {ratings['rating'].min()} - {ratings['rating'].max()}")
    print(f"   Avg rating:  {ratings['rating'].mean():.2f}")

    # 2. Train/Test split
    train_df, test_df = train_test_split(ratings, test_ratio=0.2)
    print(f"\n   Train: {len(train_df)} ratings | Test: {len(test_df)} ratings")

    # 3. Train Collaborative Filtering (on train set)
    print("\n🔧 Training Collaborative Filtering (SVD)...")
    cf = CollaborativeFilter(n_components=20)
    cf.fit(train_df)
    print("   ✅ Done")

    # 4. Train Content-Based Filtering
    print("\n🔧 Training Content-Based Filtering (TF-IDF)...")
    cb = ContentBasedFilter()
    cb.fit(items)
    print("   ✅ Done")

    # 5. Build Hybrid Recommender
    print("\n🔧 Building Hybrid Recommender (CF: 60% + CB: 40%)...")
    hybrid = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    hybrid.fit(cf, cb)
    print("   ✅ Done")

    # 6. Sample recommendations
    sample_user = train_df["user_id"].iloc[0]
    print(f"\n📋 Recommendations for user '{sample_user}':")

    user_rated = train_df[train_df["user_id"] == sample_user].sort_values("rating", ascending=False)
    print(f"\n   User's top-rated items (train):")
    for _, row in user_rated.head(3).iterrows():
        title = items.loc[items["item_id"] == row["item_id"], "title"].values[0]
        genre = items.loc[items["item_id"] == row["item_id"], "genre"].values[0]
        print(f"     ⭐ {row['rating']:.1f}  {title} ({genre})")

    print(f"\n   🔀 Hybrid recommendations:")
    for iid, score in hybrid.recommend(sample_user, train_df, top_k=5):
        title = items.loc[items["item_id"] == iid, "title"].values[0]
        genre = items.loc[items["item_id"] == iid, "genre"].values[0]
        print(f"     → {title} ({genre})  score={score:.4f}")

    print(f"\n   👥 Collaborative Filtering recommendations:")
    for iid, score in cf.recommend(sample_user, top_k=5):
        title = items.loc[items["item_id"] == iid, "title"].values[0]
        genre = items.loc[items["item_id"] == iid, "genre"].values[0]
        print(f"     → {title} ({genre})  score={score:.4f}")

    # 7. Evaluate on test set
    print("\n" + "=" * 60)
    print("  EVALUATION (on held-out test set)")
    print("=" * 60)
    summary = evaluate_model(cf, hybrid, train_df, test_df, k=5)
    for model_name, metrics in summary.items():
        print(f"\n  {model_name.upper()}:")
        for metric, value in metrics.items():
            if value is not None:
                print(f"    {metric}: {value}")

    print("\n" + "=" * 60)
    print("  ✅ Demo complete!")
    print("  💡 Run 'python app.py' to start the Flask API")
    print("=" * 60)


if __name__ == "__main__":
    main()
