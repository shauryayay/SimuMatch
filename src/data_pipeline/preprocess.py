# src/data_pipeline/preprocess.py
import pandas as pd
import os

def preprocess_athletes(input_path="data/raw/athlete_events.csv",
                        output_path="data/processed/clean_athletes.csv"):
    
    print("Preprocessing athlete data...")
    df = pd.read_csv(input_path)

    df.columns = [c.lower().strip() for c in df.columns]

    df = df.dropna(subset=["name"])
    df["name"] = df["name"].str.title()

    df["sex"] = df["sex"].str.upper().replace({
        "FEMALE": "F", "W": "F", "MALE": "M"
    })

    df = df.drop_duplicates(subset=["id"])

    for col in ["age", "height", "weight", "year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sport"] = df["sport"].fillna("Unknown Sport")
    df["event"] = df["event"].fillna("Unknown Event")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Preprocessed file saved → {output_path}")
    return df


if __name__ == "__main__":
    preprocess_athletes()
