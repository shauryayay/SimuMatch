from src.data_pipeline.preprocess import preprocess_athletes
from src.matching.vector_search import generate_athlete_embeddings
from src.matching.event_embeddings import generate_event_embeddings

def main(force_rebuild=False):
    print("Running full SimuMatch pipeline...\n")

    preprocess_athletes()
    generate_athlete_embeddings(force_rebuild=force_rebuild)
    generate_event_embeddings(force_rebuild=force_rebuild)

    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
