import json
import time
import logging
from pathlib import Path

from src.core.embeddings import EmbeddingGenerator
from src.core.retriever import VectorStore
from src.core.agent import RAGAgent  # Ensure correct import
from src.utils.document_loader import DocumentLoader

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Queries for evaluation
EVAL_QUERIES = [
    "What are the main components of the system?",
    "How does the document processing work?",
    "What's the difference between document processing and query processing?",
    "Explain quantum teleportation in this system context.",
    "How does the system load documents, store embeddings, and answer queries step by step?"
]

def main():
    logger.info("Starting Evaluation of RAG Prototype")

    # --- Build components (mirror run_demo.py) ---
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = VectorStore(collection_name="test_collection")
    agent = RAGAgent(vector_store=vector_store, embedding_generator=embedding_generator)

    # --- Ensure documents exist (if collection is empty) ---
    doc_dir = Path("sample_corpus")
    if not doc_dir.exists() or not any(doc_dir.glob("*.txt")):
        logger.error("No documents found. Please run run_demo.py first.")
        return

    # --- Process each query ---
    results = []
    for query in EVAL_QUERIES:
        start = time.time()
        response = agent.process_query(query)  # RAGAgent usually uses process_query()
        elapsed = round(time.time() - start, 2)

        results.append({
            "query": query,
            "answer": response.get("answer", ""),
            "retrieved_chunks": response.get("retrieved_chunks", []),
            "context_used": response.get("context_used", ""),
            "time_taken_sec": elapsed
        })

        logger.info(f"Query: {query}")
        logger.info(f"Answer: {response.get('answer', '')}")
        logger.info(f"Time Taken: {elapsed}s\n")

    # --- Save results ---
    output_file = Path("evaluation_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Evaluation completed. Results saved to {output_file}")

if __name__ == "__main__":
    main()
