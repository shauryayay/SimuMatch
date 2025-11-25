import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load raw Kaggle data
df = pd.read_csv("data/raw/athlete_events.csv")

# Standardize some columns (but we KEEP Sex, Sport, Event names)
df.rename(columns={
    "ID": "id",
    "Name": "name",
    "NOC": "noc"
}, inplace=True)

print("Data loaded with columns:", df.columns.tolist())

# Drop duplicate athletes by id (keep first row per athlete)
df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)


def build_athlete_profile_text(row):
    """
    Build a descriptive text used for embedding an athlete.
    Uses: name, sex, age, height, weight, team, noc, sport, event.
    """
    parts = []

    if pd.notnull(row.get("name")):
        parts.append(str(row["name"]))

    # Sex
    sex = row.get("Sex", None)
    if pd.notnull(sex):
        if str(sex).upper() == "M":
            parts.append("male")
        elif str(sex).upper() == "F":
            parts.append("female")

    # Age
    if pd.notnull(row.get("Age")):
        parts.append(f"{int(row['Age'])} years old")

    # Height / Weight
    if pd.notnull(row.get("Height")):
        parts.append(f"height {int(row['Height'])} cm")
    if pd.notnull(row.get("Weight")):
        parts.append(f"weight {int(row['Weight'])} kg")

    # Team & NOC
    if pd.notnull(row.get("Team")):
        parts.append(f"from team {row['Team']}")
    if pd.notnull(row.get("noc")):
        parts.append(f"NOC {row['noc']}")

    # Sport & Event
    if pd.notnull(row.get("Sport")):
        parts.append(f"plays {row['Sport']}")
    if pd.notnull(row.get("Event")):
        parts.append(f"competes in {row['Event']}")

    return " | ".join(parts)


df["profile_text"] = df.apply(build_athlete_profile_text, axis=1)

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

texts = df["profile_text"].tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# Save embeddings into a separate .npy file and also as a column (for inspection)
df["emb"] = embeddings.tolist()

np.save("data/athlete_vectors.npy", np.array(embeddings))
df.to_csv("data/processed/athletes_with_embeddings.csv", index=False)

print("Athlete embeddings saved!")
