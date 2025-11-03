import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv("data/raw/athlete_events.csv")

df.rename(columns={
    "Event": "event"
}, inplace=True)

# unique events only
df_events = df[["event"]].drop_duplicates()

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

texts = df_events["event"].tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

df_events["emb"] = embeddings.tolist()

np.save("data/event_vectors.npy", np.array(embeddings))
df_events.to_csv("data/processed/events_with_embeddings.csv", index=False)

print("Event embeddings saved!")
