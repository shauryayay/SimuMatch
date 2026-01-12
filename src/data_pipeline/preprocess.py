import pandas as pd
import os
from src.utils.schema import validate_schema


def preprocess_athletes(
    input_path="data/raw/athlete_events.csv",
    output_path="data/processed/clean_athletes.csv"
):
    print("Preprocessing athlete data...")

    # Load raw data
    df = pd.read_csv(input_path)

    # Normalize column names FIRST
    df.columns = [c.lower().strip() for c in df.columns]

    # Validate normalized schema
    validate_schema(
        df,
        ["id", "name", "sex", "sport", "event"],
        df_name="normalized athlete_events.csv"
    )

    # Drop rows with no athlete name
    df = df.dropna(subset=["name"])

    # Clean names
    df["name"] = df["name"].str.title()

    # Normalize sex values
    df["sex"] = df["sex"].astype(str).str.upper().replace({
        "FEMALE": "F",
        "MALE": "M",
        "W": "F"
    })

    # Remove duplicate athletes (keep first occurrence)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    else:
        df = df.drop_duplicates(subset=["name"])

    # Ensure numeric columns
    for col in ["age", "height", "weight", "year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing sport/event
    df["sport"] = df["sport"].fillna("Unknown Sport")
    df["event"] = df["event"].fillna("Unknown Event")

    # Save processed data
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Preprocessed file saved → {output_path}")
    return df


if __name__ == "__main__":
    preprocess_athletes()
