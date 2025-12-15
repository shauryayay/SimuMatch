# src/matching/event_embeddings.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_event_embeddings(
    input_path="data/processed/clean_athletes.csv",
    output_csv="data/processed/events_with_embeddings.csv",
    output_npy="data/event_vectors.npy"
):
    print("Generating event embeddings...")

    df = pd.read_csv(input_path)

    event_meta = (
        df.groupby("event")
          .agg({
              "sport": "first",
              "sex": lambda x: x.mode().iloc[0] if not x.mode().empty else None
          })
          .reset_index()
    )

    event_meta.rename(columns={"sex": "event_sex"}, inplace=True)

    event_meta["event_text"] = (
        event_meta["sport"].fillna("") + " "
        + event_meta["event_sex"].map(lambda x: "men " if x=="M" else ("women " if x=="F" else ""))
        + event_meta["event"].fillna("")
    )

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    texts = event_meta["event_text"].tolist()

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    event_meta["emb"] = embeddings.tolist()

    np.save(output_npy, embeddings)
    event_meta.to_csv(output_csv, index=False)

    print("Event embeddings saved!")
    return event_meta


if __name__ == "__main__":
    generate_event_embeddings()
