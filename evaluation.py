"""
Evaluation metrics for recommendation engines.
Uses proper train/test split — trains on 80% of ratings, tests on held-out 20%.
Computes Precision@k, Recall@k, and RMSE.
"""
import numpy as np
import pandas as pd


def train_test_split(ratings_df, test_ratio=0.2, seed=42):
    """Split ratings into train/test per user."""
    train_list, test_list = [], []
    for uid, group in ratings_df.groupby("user_id"):
        n = len(group)
        if n < 5:
            train_list.append(group)
            continue
        test = group.sample(frac=test_ratio, random_state=seed)
        train = group.drop(test.index)
        train_list.append(train)
        test_list.append(test)
    return pd.concat(train_list), pd.concat(test_list)


def precision_at_k(recommended, relevant, k=5):
    """Fraction of recommended items in top-k that are relevant."""
    rec_k = [iid for iid, _ in recommended[:k]]
    relevant_set = set(relevant)
    if len(rec_k) == 0:
        return 0.0
    hits = len(set(rec_k) & relevant_set)
    return hits / k


def recall_at_k(recommended, relevant, k=5):
    """Fraction of relevant items that appear in top-k recommendations."""
    rec_k = [iid for iid, _ in recommended[:k]]
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0.0
    hits = len(set(rec_k) & relevant_set)
    return hits / len(relevant_set)


def evaluate_model(cf_model, hybrid_model, train_df, test_df, k=5):
    """
    Evaluate CF and Hybrid models using a train/test split.
    Relevant items = items the user rated >= 3.5 in the TEST set (ground truth).
    """
    results = {"collaborative": {"precision": [], "recall": []},
               "hybrid": {"precision": [], "recall": []}}

    test_users = test_df["user_id"].unique()

    for uid in test_users:
        # Ground truth: highly-rated items in test set
        user_test = test_df[test_df["user_id"] == uid]
        relevant = user_test[user_test["rating"] >= 3.5]["item_id"].tolist()

        if len(relevant) < 1:
            continue

        # CF recommendations (trained on train_df, so test items are unseen)
        cf_recs = cf_model.recommend(uid, top_k=k)
        results["collaborative"]["precision"].append(precision_at_k(cf_recs, relevant, k))
        results["collaborative"]["recall"].append(recall_at_k(cf_recs, relevant, k))

        # Hybrid recommendations
        hy_recs = hybrid_model.recommend(uid, train_df, top_k=k)
        results["hybrid"]["precision"].append(precision_at_k(hy_recs, relevant, k))
        results["hybrid"]["recall"].append(recall_at_k(hy_recs, relevant, k))

    # Aggregate
    summary = {}
    for model_name in results:
        p = np.mean(results[model_name]["precision"]) if results[model_name]["precision"] else 0
        r = np.mean(results[model_name]["recall"]) if results[model_name]["recall"] else 0
        summary[model_name] = {"precision@k": round(float(p), 4), "recall@k": round(float(r), 4)}

    # RMSE for CF on test set
    rmse = cf_model.evaluate(test_df)
    summary["collaborative"]["rmse"] = round(rmse, 4) if rmse else None

    return summary
