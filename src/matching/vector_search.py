import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv("data/raw/athlete_events.csv")

# Rename columns to standard names for internal use
df.rename(columns={
    "ID": "id",
    "Name": "name",
    "Sport": "sport",
    "Event": "event",
    "NOC": "country"
}, inplace=True)

print("Data loaded with columns:", df.columns)

# Drop duplicates so we don't embed same athlete multiple times
df = df.drop_duplicates(subset=["id"])

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

texts = df["name"].tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

df["emb"] = embeddings.tolist()

np.save("data/athlete_vectors.npy", np.array(embeddings))
df.to_csv("data/processed/athletes_with_embeddings.csv", index=False)

print("Embeddings saved.")
