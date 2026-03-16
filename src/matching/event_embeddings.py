# src/matching/event_embeddings.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def generate_event_embeddings(
    input_path="data/processed/clean_athletes.csv",
    output_csv="data/processed/events_with_embeddings.csv",
    output_npy="data/event_vectors.npy",
    force_rebuild=False
):
    if (
        not force_rebuild
        and os.path.exists(output_csv)
        and os.path.exists(output_npy)
    ):
        print("⚡ Using cached event embeddings")
        return pd.read_csv(output_csv)

    print("Generating event embeddings...")

    df = pd.read_csv(input_path)

    # (keep your existing event_meta + event_text logic)

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(
        event_meta["event_text"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )

    event_meta["emb"] = embeddings.tolist()
    np.save(output_npy, embeddings)
    event_meta.to_csv(output_csv, index=False)

    print("Event embeddings saved!")
    return event_meta


if __name__ == "__main__":
    generate_event_embeddings()
