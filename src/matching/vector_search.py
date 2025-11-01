from sentence_transformers import SentenceTransformer, util
import pandas as pd

model = SentenceTransformer("all-mpnet-base-v2")

def embed_text(text):
    return model.encode(text)

if __name__ == "__main__":
    df = pd.read_csv("data/processed/athletes_clean.csv")
    df["emb"] = df["name"].apply(lambda x: embed_text(x))
    print("Embeddings generated")
