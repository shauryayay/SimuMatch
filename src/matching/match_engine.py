import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
from src.utils.schema import validate_schema


# -----------------------------
# Load data
# -----------------------------
athlete_df = pd.read_csv("data/processed/athletes_with_embeddings.csv")
validate_schema(
    athlete_df,
    ["id", "name", "sex", "sport", "event", "emb"],
    df_name="athletes_with_embeddings.csv"
)

event_df   = pd.read_csv("data/processed/events_with_embeddings.csv")


# Load embeddings
ath_embeddings  = np.load("data/athlete_vectors.npy", allow_pickle=True)
event_embeddings = np.load("data/event_vectors.npy", allow_pickle=True)

# Attach embeddings to dataframes
athlete_df["emb"] = list(ath_embeddings)
event_df["emb"]   = list(event_embeddings)

# Normalize ALL column names to lowercase internally
athlete_df.columns = [c.lower() for c in athlete_df.columns]
event_df.columns   = [c.lower() for c in event_df.columns]

# -----------------------------
# Helper: Robust Name Matching
# -----------------------------
def _clean_name(n: str):
    n = str(n).lower()
    n = re.sub(r"[^a-z ]", " ", n)
    return " ".join(n.split())

def _split_first_last(full_name: str):
    full_name = _clean_name(full_name)
    tokens = full_name.split()
    if not tokens:
        return None, None
    return tokens[0], tokens[-1]

athlete_df["_first"] = athlete_df["name"].apply(lambda x: _split_first_last(x)[0])
athlete_df["_last"]  = athlete_df["name"].apply(lambda x: _split_first_last(x)[1])

def find_athlete_index(query: str):
    """
    Hierarchy:
    1) exact full-name match
    2) first+last name match
    3) substring match
    4) all tokens present
    """

    q_clean = _clean_name(query)
    tokens  = q_clean.split()

    # exact
    exact = athlete_df[athlete_df["name"].str.lower() == q_clean]
    if len(exact) > 0:
        return exact.index[0]

    # first + last
    if len(tokens) >= 2:
        qf, ql = tokens[0], tokens[-1]
        fl = athlete_df[
            (athlete_df["_first"] == qf) &
            (athlete_df["_last"]  == ql)
        ]
        if len(fl) > 0:
            return fl.index[0]

    # substring
    substr = athlete_df[athlete_df["name"].str.lower().str.contains(q_clean)]
    if len(substr) > 0:
        return substr.index[0]

    # all tokens must appear
    if tokens:
        mask = pd.Series(True, index=athlete_df.index)
        for t in tokens:
            mask &= athlete_df["name"].str.lower().str.contains(t)
        all_toks = athlete_df[mask]
        if len(all_toks) > 0:
            return all_toks.index[0]

    return None

# -----------------------------
# Main Recommendation Function
# -----------------------------
def recommend_events(name: str, top_k=5):
    idx = find_athlete_index(name)
    if idx is None:
        return f"Athlete '{name}' not found."

    athlete = athlete_df.loc[idx]

    # Debug info (now casing-proof)
    debug_display = {
        "name": athlete.get("name"),
        "sport": athlete.get("sport"),
        "sex": athlete.get("sex")
    }
    print("DEBUG athlete match:", debug_display)

    # Get vector
    athlete_vector = np.array(athlete["emb"]).reshape(1, -1)

    # Extract metadata
    sport = athlete.get("sport", None)
    sex   = athlete.get("sex", None)

    filtered = event_df.copy()

    # Sport filter
    if pd.notnull(sport):
        filtered = filtered[filtered["sport"] == sport]

    # Sex filter
    if pd.notnull(sex) and "event_sex" in filtered.columns:
        filtered = filtered[filtered["event_sex"] == sex]

    # fallback #1 — if filtering got too strict
    if len(filtered) == 0 and pd.notnull(sport):
        filtered = event_df[event_df["sport"] == sport]

    # fallback #2 — just recommend anything
    if len(filtered) == 0:
        filtered = event_df.copy()

    # Compute similarities
    matrix = np.vstack(filtered["emb"].to_list())
    sims   = cosine_similarity(athlete_vector, matrix)[0]

    # Pick top-k
    top_idx = sims.argsort()[::-1][:top_k]
    top = filtered.iloc[top_idx].copy()
    top["similarity"] = sims[top_idx]

    return top[["sport", "event", "event_sex", "similarity"]].reset_index(drop=True)

# For CLI testing
if __name__ == "__main__":
    print(recommend_events("Usain Bolt", top_k=5))
