# test_agent.py
from src.core.agent import RAGAgent
from src.core.retriever import VectorStore
from src.core.embeddings import EmbeddingGenerator
from src.utils.logging import setup_logging
from loguru import logger

def test_rag():
    # Setup logging
    setup_logging()
    logger.info("Starting RAG system test")

    try:
        # Initialize components
        vector_store = VectorStore(collection_name="test_collection")
        embedding_generator = EmbeddingGenerator()
        rag_agent = RAGAgent(vector_store, embedding_generator)
        
        # Test queries
        test_queries = [
        "What is this document about?",
        "What is the capital of India?",
        "Who is the Prime Minister of India?"
        ]


        # Process queries
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            
            response = rag_agent.process_query(query)
            
            print(f"\nQuery: {query}")
            print(f"Agent Response: {response['answer']}")
            print("-" * 50)

        logger.info("Test completed successfully")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_rag()