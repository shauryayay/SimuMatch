import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

df = pd.read_csv("data/raw/athlete_events.csv")

# We need Sport, Event, Sex from raw data
# (These columns exist in the Kaggle file even if your profile view hid them to the right)
event_meta = (
    df.groupby("Event")
      .agg({
          "Sport": "first",
          "Sex": lambda x: x.mode().iloc[0] if not x.mode().empty else None
      })
      .reset_index()
)

event_meta.rename(columns={
    "Event": "event",
    "Sport": "sport",
    "Sex": "event_sex"
}, inplace=True)


def build_event_text(row):
    parts = []

    if pd.notnull(row["sport"]):
        parts.append(str(row["sport"]))

    sex = row["event_sex"]
    if pd.notnull(sex):
        if str(sex).upper() == "M":
            parts.append("men")
        elif str(sex).upper() == "F":
            parts.append("women")

    if pd.notnull(row["event"]):
        parts.append(str(row["event"]))

    return " ".join(parts)


event_meta["event_text"] = event_meta.apply(build_event_text, axis=1)

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

texts = event_meta["event_text"].tolist()
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

event_meta["emb"] = embeddings.tolist()

np.save("data/event_vectors.npy", np.array(embeddings))
event_meta.to_csv("data/processed/events_with_embeddings.csv", index=False)

print("Event embeddings saved!")

