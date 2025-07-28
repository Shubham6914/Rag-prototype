# src/main.py

import typer
from pathlib import Path
from loguru import logger

from src.utils.logging import setup_logging
from src.utils.document_loader import DocumentLoader
from src.utils.chunking import TextChunker
from src.core.embeddings import EmbeddingGenerator
from src.core.retriever import VectorStore
from src.core.agent import RAGAgent

# Create CLI app
app = typer.Typer()

def initialize_components():
    """Initialize all system components"""
    try:
        doc_loader = DocumentLoader()
        chunker = TextChunker(chunk_size=300, chunk_overlap=50)
        vector_store = VectorStore(collection_name="test_collection")
        embedding_generator = EmbeddingGenerator()
        rag_agent = RAGAgent(vector_store, embedding_generator)
        return doc_loader, chunker, embedding_generator, vector_store, rag_agent
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

def check_document_processed(vector_store, file_path: Path) -> bool:
    """Check if a document has already been processed (based on metadata)"""
    try:
        results = vector_store.retrieve(
            query_embedding=[0.0] * 384,  # Adjust dimension if needed
            limit=1,
            metadata_filter={"source": str(file_path)}
        )
        return len(results) > 0
    except Exception as e:
        logger.error(f"Error checking document status: {e}")
        return False

@app.command()
def ingest(doc_folder: str = typer.Argument(..., help="Folder containing documents to ingest")):
    """Ingest and index documents from a folder"""
    setup_logging()
    doc_loader = DocumentLoader()
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore(collection_name="test_collection")

    folder = Path(doc_folder)
    files = list(folder.glob("*.txt")) + list(folder.glob("*.md")) + list(folder.glob("*.pdf"))

    if not files:
        logger.warning("No valid documents found in the folder.")
        raise typer.Exit()

    for file in files:
        logger.info(f"Processing document: {file.name}")
        try:
            if check_document_processed(vector_store, file):
                logger.info(f"{file.name} already processed. Skipping.")
                continue

            content = doc_loader.load_document(file)
            chunks = chunker.chunk_text(content)
            embeddings = embedding_generator.generate_embeddings(chunks)

            vector_store.store_documents(
                texts=chunks,
                embeddings=embeddings,
                metadata=[{"source": str(file)} for _ in chunks]
            )

            logger.info(f"{file.name} ingested successfully.")
        except Exception as e:
            logger.error(f"Failed to process {file.name}: {e}")

@app.command()
def process_query(query: str):
    """Process a single query"""
    setup_logging()
    _, _, _, _, rag_agent = initialize_components()
    
    logger.info(f"Processing query: {query}")
    response = rag_agent.process_query(query)
    print(f"\nAgent Response: {response['answer']}")

@app.command()
def run_tests():
    """Run test scenarios"""
    setup_logging()
    _, _, _, _, rag_agent = initialize_components()
    
    test_scenarios = [
        "What is the main purpose of the system described in the document?",
        "How is the document processing implemented?",
        "What are the different components mentioned?",
        "What is quantum physics?",  # Edge case
        "Explain how the system processes and retrieves information?"
    ]

    for query in test_scenarios:
        logger.info(f"Testing query: {query}")
        response = rag_agent.process_query(query)
        print(f"\nQuery: {query}")
        print(f"Agent Response: {response['answer']}")
        print("-" * 50)

if __name__ == "__main__":
    app()
