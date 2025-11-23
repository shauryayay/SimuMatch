import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load processed data
athlete_df = pd.read_csv("data/processed/athletes_with_embeddings.csv")
event_df = pd.read_csv("data/processed/events_with_embeddings.csv")

# Load embeddings
ath_embeddings = np.load("data/athlete_vectors.npy", allow_pickle=True)
event_embeddings = np.load("data/event_vectors.npy", allow_pickle=True)

# Attach embeddings to dataframes (aligned by row order)
athlete_df["emb"] = list(ath_embeddings)
event_df["emb"] = list(event_embeddings)

import re

# Decide which column is the name column
NAME_COL = "name" if "name" in athlete_df.columns else "Name"

# Clean names and precompute first / last name for robust matching
def _split_first_last(full_name: str):
    full_name = str(full_name).lower()
    # keep only letters and spaces
    full_name = re.sub(r"[^a-z ]", " ", full_name)
    tokens = [t for t in full_name.split() if t]
    if not tokens:
        return None, None
    return tokens[0], tokens[-1]

firsts = []
lasts = []
for n in athlete_df[NAME_COL].astype(str):
    f, l = _split_first_last(n)
    firsts.append(f)
    lasts.append(l)

athlete_df["_first_name"] = firsts
athlete_df["_last_name"] = lasts



def find_athlete_index(name: str):
    """
    Robust athlete name matcher:
    1. exact match on full name
    2. match on first + last name (e.g. 'usain bolt' -> 'Usain St. Leo Bolt')
    3. full substring match
    4. ALL tokens must appear (AND)
    """
    name = name.lower().strip()

    name_col = "name" if "name" in athlete_df.columns else "Name"
    names = athlete_df[name_col].astype(str).str.lower()

    # 1) exact match (full string)
    exact = athlete_df[names == name]
    if len(exact) > 0:
        return exact.index[0]

    # Prepare query tokens
    import re
    cleaned = re.sub(r"[^a-z ]", " ", name)
    tokens = [t for t in cleaned.split() if t]

    # 2) first + last name match (best for cases like 'Usain Bolt')
    if len(tokens) >= 2:
        q_first, q_last = tokens[0], tokens[-1]
        mask_fl = (athlete_df["_first_name"] == q_first) & (athlete_df["_last_name"] == q_last)
        fl_matches = athlete_df[mask_fl]
        if len(fl_matches) > 0:
            return fl_matches.index[0]

    # 3) full substring
    contains = athlete_df[names.str.contains(name, na=False)]
    if len(contains) > 0:
        return contains.index[0]

    # 4) ALL tokens must be present (AND)
    if tokens:
        mask = pd.Series(True, index=names.index)
        for t in tokens:
            mask &= names.str.contains(t, na=False)
        all_tokens = athlete_df[mask]
        if len(all_tokens) > 0:
            return all_tokens.index[0]

    # nothing reasonable found
    return None

def recommend_events(name: str, top_k: int = 5):
    idx = find_athlete_index(name)
    if idx is None:
        return f"Athlete '{name}' not found."
    
    print("DEBUG matched athlete:", athlete_df.loc[idx, ["name", "Sport", "Sex"]].to_dict())

    athlete_row = athlete_df.loc[idx]

    # Embed
    athlete_vector = np.array(athlete_row["emb"]).reshape(1, -1)

    # Athlete sport & sex from original columns
    sport = athlete_row.get("Sport", None)
    sex = athlete_row.get("Sex", None)

    filtered = event_df.copy()

    # 1) Filter by sport if possible
    if pd.notnull(sport) and "sport" in filtered.columns:
        filtered = filtered[filtered["sport"] == sport]

    # 2) Filter by sex if possible
    if pd.notnull(sex) and "event_sex" in filtered.columns:
        filtered = filtered[filtered["event_sex"] == sex]

    # Fallbacks if filter is too strict
    if filtered.empty and pd.notnull(sport) and "sport" in event_df.columns:
        filtered = event_df[event_df["sport"] == sport]

    if filtered.empty:
        filtered = event_df.copy()

    # Build matrix for filtered events
    event_matrix = np.vstack(filtered["emb"].to_list())
    sims = cosine_similarity(athlete_vector, event_matrix)[0]

    top_indices = sims.argsort()[::-1][:top_k]
    top_events = filtered.iloc[top_indices].copy()
    top_events["similarity"] = sims[top_indices]

    return top_events[["sport", "event", "event_sex", "similarity"]].reset_index(drop=True)


if __name__ == "__main__":
    print(recommend_events("Usain St. Leo Bolt", top_k=5))

