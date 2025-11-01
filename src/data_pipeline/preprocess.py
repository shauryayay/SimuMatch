import pandas as pd

def clean_athlete_data(input_path="data/raw/athletes.csv"):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=["name"]) 
    df["name"] = df["name"].str.title()
    df.to_csv("data/processed/athletes_clean.csv", index=False)
    print("✨ Clean data saved")

if __name__ == "__main__":
    clean_athlete_data()
