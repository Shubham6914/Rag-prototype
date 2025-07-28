from pathlib import Path
from loguru import logger
from src.utils.logging import setup_logging
from src.utils.document_loader import DocumentLoader
from src.utils.chunking import TextChunker
from src.core.embeddings import EmbeddingGenerator
from src.core.retriever import VectorStore
from src.core.agent import RAGAgent
import time

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
    }
]

def setup_demo_environment():
    try:
        sample_corpus = Path("sample_corpus")
        text_files = list(sample_corpus.glob("*.txt")) + list(sample_corpus.glob("*.md"))

        if not text_files:
            logger.error("No text files found in sample_corpus directory")
            return None

        logger.info(f"Found {len(text_files)} documents in sample_corpus")
        for file in text_files:
            logger.info(f"Available document: {file}")

        demo_file = text_files[0]
        logger.info(f"Using document for demo: {demo_file}")
        return demo_file

    except Exception as e:
        logger.error(f"Demo setup failed: {e}")
        return None

def check_document_processed(vector_store, file_path: Path) -> bool:
    try:
        results = vector_store.retrieve(
            query_embedding=[0.0] * 384,  # embedding dimension should match your model, adjust if needed
            limit=1,
            metadata_filter={"source": str(file_path)}
        )
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking document status: {e}")
        return False

def process_documents(doc_loader, chunker, embedding_generator, vector_store, file_path: Path):
    try:
        if check_document_processed(vector_store, file_path):
            logger.info(f"Document {file_path} already processed, skipping...")
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
    setup_logging()
    logger.info("Starting RAG System Demo")

    try:
        print("\n=== Initializing Components ===")
        doc_loader = DocumentLoader()
        chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(collection_name="test_collection")

        rag_agent = RAGAgent(vector_store, embedding_generator)

        demo_file = setup_demo_environment()
        if not demo_file:
            raise Exception("No documents found")

        print("\n=== Checking Document Status ===")
        if process_documents(doc_loader, chunker, embedding_generator, vector_store, demo_file):
            print("Document processing complete")
        else:
            raise Exception("Document processing failed")

        print("\n=== Document Processing Complete ===")
        print("\n=== Running Test Scenarios ===")
        print("\n" + "="*50)
        print("RAG System Evaluation Results")
        print("="*50)

        for scenario in TEST_SCENARIOS:
            print(f"\nTest Type: {scenario['type']}")
            print(f"Query: {scenario['query']}")
            print("-"*30)

            response = rag_agent.process_query(scenario['query'])

            # Instead of re-calling _format_prompt and generator, use response directly:
            print("Agent Response:")
            print(response.get("answer", "No answer returned.").strip())

            print(f"\nContext Used:")
            if response.get("context_used"):
                print(response["context_used"][:200] + "...")
            else:
                print("No context used.")

            print("="*50)
            time.sleep(1)

        print("\nDemo Summary:")
        print("✓ Document Processing")
        print("✓ Vector Storage")
        print("✓ Query Processing")
        print("✓ Response Generation")
        logger.info("Demo completed successfully")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nError during demo: {e}")
        raise

if __name__ == "__main__":
    run_demo()
