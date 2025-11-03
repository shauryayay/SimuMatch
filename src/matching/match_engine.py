import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

athlete_df = pd.read_csv("data/processed/athletes_with_embeddings.csv")
event_df = pd.read_csv("data/processed/events_with_embeddings.csv")

ath_embeddings = np.load("data/athlete_vectors.npy", allow_pickle=True)
event_embeddings = np.load("data/event_vectors.npy", allow_pickle=True)

athlete_df["emb"] = ath_embeddings.tolist()
event_df["emb"] = event_embeddings.tolist()


def find_athlete_index(name):
    name = name.lower().strip()

    # exact match
    exact = athlete_df[athlete_df['name'].str.lower() == name]
    if len(exact) > 0:
        return exact.index[0]

    # contains match
    contains = athlete_df[athlete_df['name'].str.lower().str.contains(name)]
    if len(contains) > 0:
        return contains.index[0]

    # split name parts and match
    parts = name.split()
    regex = "|".join(parts)
    fuzzy = athlete_df[athlete_df['name'].str.lower().str.contains(regex)]
    if len(fuzzy) > 0:
        return fuzzy.index[0]

    return None


def recommend_events(name, top_k=5):
    idx = find_athlete_index(name)
    if idx is None:
        return "Athlete not found."

    athlete_vector = np.array(athlete_df.loc[idx, "emb"]).reshape(1, -1)

    # compute sim with all events
    sims = cosine_similarity(athlete_vector, np.vstack(event_embeddings))[0]
    top_indices = sims.argsort()[::-1][:top_k]

    return event_df.iloc[top_indices][["event"]]
