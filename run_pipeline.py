from src.data_pipeline.preprocess import preprocess_athletes
from src.matching.vector_search import generate_athlete_embeddings
from src.matching.event_embeddings import generate_event_embeddings

def main():
    print("Running full SimuMatch pipeline...\n")

    preprocess_athletes()
    generate_athlete_embeddings()
    generate_event_embeddings()

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
