import pandas as pd
from neo4j_connection import get_driver

def push_athletes():
    df = pd.read_csv("data/processed/athletes_clean.csv")
    driver = get_driver()
    with driver.session() as session:
        for _, row in df.iterrows():
            session.run(
                """
                MERGE (a:Athlete {id:$id})
                SET a.name=$name, a.country=$country
                """,
                id=row["id"], name=row["name"], country=row.get("country","Unknown")
            )
    print("Graph nodes inserted!")

if __name__ == "__main__":
    push_athletes()
