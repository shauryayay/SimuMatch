# src/matching/vector_search.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from src.utils.schema import validate_schema


def generate_athlete_embeddings(
        input_path="data/processed/clean_athletes.csv",
        output_csv="data/processed/athletes_with_embeddings.csv",
        output_npy="data/athlete_vectors.npy"):

    print("Generating athlete embeddings...")
    df = pd.read_csv(input_path)

    validate_schema(
    df,
    ["id", "name", "sex", "sport", "event"],
    df_name="clean_athletes.csv"
)


    # Vectorized text creation (fast)
    def safe(x): 
        return "" if pd.isna(x) else str(x)

    df["profile_text"] = (
          df["name"].map(safe)
        + " | " + df["sex"].map(lambda x: "male" if x=="M" else ("female" if x=="F" else ""))
        + " | " + df["age"].map(lambda x: f"{int(x)} years old" if pd.notna(x) else "")
        + " | " + df["team"].map(lambda x: f"team {x}" if pd.notna(x) else "")
        + " | plays " + df["sport"].map(safe)
        + " | competes in " + df["event"].map(safe)
    )

    model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    texts = df["profile_text"].tolist()

    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    df["emb"] = embeddings.tolist()

    np.save(output_npy, embeddings)
    df.to_csv(output_csv, index=False)

    print("Athlete embeddings saved!")
    return df


if __name__ == "__main__":
    generate_athlete_embeddings()
