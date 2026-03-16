# src/matching/vector_search.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

def generate_athlete_embeddings(
    input_path="data/processed/clean_athletes.csv",
    output_csv="data/processed/athletes_with_embeddings.csv",
    output_npy="data/athlete_vectors.npy",
    force_rebuild=False
):
    # CACHE CHECK
    if (
        not force_rebuild
        and os.path.exists(output_csv)
        and os.path.exists(output_npy)
    ):
        print("⚡ Using cached athlete embeddings")
        return pd.read_csv(output_csv)

    print("Generating athlete embeddings...")

    df = pd.read_csv(input_path)

    # (keep your existing vectorized profile_text logic here)

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    embeddings = model.encode(
        df["profile_text"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )

    df["emb"] = embeddings.tolist()
    np.save(output_npy, embeddings)
    df.to_csv(output_csv, index=False)

    print("Athlete embeddings saved!")
    return df


if __name__ == "__main__":
    generate_athlete_embeddings()
