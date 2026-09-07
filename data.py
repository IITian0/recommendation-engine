"""
Sample data generation for the recommendation engine.
Generates synthetic users, items (movies), and ratings.
"""
import pandas as pd
import numpy as np

def generate_sample_data(n_users=200, n_items=100, n_ratings=3000, seed=42):
    np.random.seed(seed)

    # --- Users ---
    user_ids = [f"U{1000+i}" for i in range(n_users)]
    users = pd.DataFrame({
        "user_id": user_ids,
        "age": np.random.randint(18, 60, n_users),
        "gender": np.random.choice(["M", "F"], n_users),
    })

    # --- Items (movies) ---
    genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance", "Thriller"]
    item_ids = [f"M{100+i}" for i in range(n_items)]
    items = pd.DataFrame({
        "item_id": item_ids,
        "title": [f"Movie_{i}" for i in range(n_items)],
        "genre": np.random.choice(genres, n_items),
        "year": np.random.randint(1990, 2024, n_items),
    })

    # --- Ratings ---
    # Create a latent preference: each user has a favourite genre
    user_fav_genre = {uid: np.random.choice(genres) for uid in user_ids}

    ratings_list = []
    for _ in range(n_ratings):
        uid = np.random.choice(user_ids)
        iid = np.random.choice(item_ids)
        item_genre = items.loc[items["item_id"] == iid, "genre"].values[0]

        # Users rate favourite-genre items higher
        base = 3.0
        if item_genre == user_fav_genre[uid]:
            base += 1.5
        rating = round(np.clip(base + np.random.normal(0, 0.8), 1, 5), 1)
        ratings_list.append({"user_id": uid, "item_id": iid, "rating": rating})

    ratings = pd.DataFrame(ratings_list).drop_duplicates(subset=["user_id", "item_id"])
    return users, items, ratings


if __name__ == "__main__":
    users, items, ratings = generate_sample_data()
    print(f"Users:    {len(users)}")
    print(f"Items:    {len(items)}")
    print(f"Ratings:  {len(ratings)}")
    print("\nSample ratings:")
    print(ratings.head(10))
