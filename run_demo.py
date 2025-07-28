from pathlib import Path
import time
from loguru import logger

from src.utils.logging import setup_logging
from src.utils.document_loader import DocumentLoader
from src.utils.chunking import TextChunker
from src.core.embeddings import EmbeddingGenerator
from src.core.retriever import VectorStore
from src.core.agent import RAGAgent
from typing import Optional

# Test scenarios for evaluation
TEST_SCENARIOS = [
    {
        "type": "Factual",
        "query": "What are the main components of the system?",
        "expected": "Should list the four main components"
    },
    {
        "type": "Procedural",
        "query": "How does the document processing work?",
        "expected": "Should explain document loading and chunking"
    },
    {
        "type": "Comparative",
        "query": "What's the difference between document processing and query processing?",
        "expected": "Should compare these two components"
    },
    {
        "type": "Edge Case",
        "query": "Explain quantum teleportation in this system context.",
        "expected": "Should gracefully handle irrelevant query"
    },
    {
        "type": "Complex",
        "query": "How does the system load documents, store embeddings, and answer queries step by step?",
        "expected": "Should provide a multi-step reasoning answer"
    },
]


def setup_demo_environment() -> Optional[Path]:
    """Locate a sample document for testing"""
    try:
        sample_corpus = Path("sample_corpus")
        text_files = list(sample_corpus.glob("*.txt")) + list(sample_corpus.glob("*.md")) + list(sample_corpus.glob("*.pdf"))

        if not text_files:
            logger.error("No documents found in sample_corpus directory")
            return None  # Now valid

        logger.info(f"Found {len(text_files)} documents in sample_corpus")
        for file in text_files:
            logger.info(f"Available document: {file}")

        demo_file = text_files[0]
        logger.info(f"Using document for demo: {demo_file}")
        return demo_file

    except Exception as e:
        logger.error(f"Demo setup failed: {e}")
        return None


def check_document_processed(vector_store: VectorStore, file_path: Path) -> bool:
    """Check if a document has already been indexed (via metadata filter)"""
    try:
        results = vector_store.retrieve(
            query_embedding=[0.0] * 384,  # Dummy embedding just to trigger metadata filter
            limit=1,
            metadata_filter={"source": str(file_path)}
        )
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking document status: {e}")
        return False


def process_documents(doc_loader, chunker, embedding_generator, vector_store, file_path: Path):
    """Load, chunk, embed, and store document into Qdrant"""
    try:
        if check_document_processed(vector_store, file_path):
            logger.info(f"Document {file_path} already processed, skipping ingestion.")
            return True

        logger.info(f"Processing new document: {file_path}")
        content = doc_loader.load_document(file_path)
        chunks = chunker.chunk_text(content)
        logger.info(f"Chunked text into {len(chunks)} chunks")

        for i, chunk in enumerate(chunks[:2]):
            logger.debug(f"Chunk {i+1}: {chunk[:100]}...")

        embeddings = embedding_generator.generate_embeddings(chunks)

        vector_store.store_documents(
            texts=chunks,
            embeddings=embeddings,
            metadata=[{"source": str(file_path)} for _ in chunks]
        )

        logger.info(f"Successfully processed document: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        return False


def run_demo():
    """Run the full demo with ingestion and evaluation tests"""
    setup_logging()
    logger.info("Starting RAG Prototype Demo")

    try:
        print("\n=== Initializing Components ===")
        doc_loader = DocumentLoader()
        chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(collection_name="test_collection")

        # Use flan-t5-base for faster inference (flan-t5-large is very slow on CPU)
        rag_agent = RAGAgent(vector_store, embedding_generator, model_name="google/flan-t5-base")

        demo_file = setup_demo_environment()
        if not demo_file:
            raise Exception("No documents found")

        print("\n=== Checking Document Status ===")
        if process_documents(doc_loader, chunker, embedding_generator, vector_store, demo_file):
            print("Document processing complete")
        else:
            raise Exception("Document processing failed")

        print("\n=== Running Evaluation Scenarios ===")
        print("\n" + "=" * 50)
        print("RAG System Evaluation Results")
        print("=" * 50)

        for scenario in TEST_SCENARIOS:
            print(f"\nTest Type: {scenario['type']}")
            print(f"Query: {scenario['query']}")
            print("-" * 30)

            start_time = time.time()
            response = rag_agent.process_query(scenario['query'])
            elapsed = time.time() - start_time

            logger.info(f"Query processed in {elapsed:.2f}s")
            logger.info(f"Agent Response: {response}")

            print(f"Agent Response:\n{response.get('answer', 'No response')}")
            print(f"\nContext Used:\n{response.get('context_used', 'Not available')}")

            print(f"\nResponse Time: {elapsed:.2f}s")
            print("=" * 50)
            time.sleep(1)

        print("\nDemo Summary:")
        print("✓ Document Processing")
        print("✓ Vector Storage (Qdrant)")
        print("✓ Query Processing")
        print("✓ Answer Generation")
        logger.info("Demo completed successfully")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nError during demo: {e}")
        raise


if __name__ == "__main__":
    run_demo()
