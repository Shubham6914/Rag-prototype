from src.utils.document_loader import DocumentLoader
from src.utils.chunking import TextChunker
from src.core.embeddings import EmbeddingGenerator
from src.core.retriever import VectorStore
from src.utils.logging import setup_logging
from pathlib import Path
import time

def test_complete_pipeline():
    # Setup logging
    setup_logging()
    
    try:
        # 1. Initialize components
        print("\n1. Initializing components...")
        doc_loader = DocumentLoader()
        chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        embedding_generator = EmbeddingGenerator()
        vector_store = VectorStore(collection_name="test_collection")
        
        # 2. Load a test document
        print("\n2. Loading test document...")
        # Assuming you have a test.txt file in sample_corpus
        test_file = Path("sample_corpus/test.txt")
        content = doc_loader.load_document(test_file)
        print(f"Loaded document with {len(content)} characters")
        
        # 3. Chunk the document
        print("\n3. Chunking document...")
        chunks = chunker.chunk_text(content)
        print(f"Created {len(chunks)} chunks")
        
        # 4. Generate embeddings
        print("\n4. Generating embeddings...")
        embeddings = embedding_generator.generate_embeddings(chunks)
        print(f"Generated embeddings of shape: {len(embeddings)}x{len(embeddings[0])}")
        
        # 5. Store in vector database
        print("\n5. Storing in Qdrant...")
        vector_store.store_documents(
            texts=chunks,
            embeddings=embeddings,
            metadata=[{"source": str(test_file)} for _ in chunks]
        )
        
        # 6. Test retrieval
        print("\n6. Testing retrieval...")
        test_query = chunks[0]  # Use first chunk as test query
        query_embedding = embedding_generator.generate_embeddings([test_query])[0]
        results = vector_store.retrieve(query_embedding, limit=2)
        
        print("\nRetrieval Results:")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text'][:200]}...")
            
        print("\nPipeline test completed successfully!")
        
    except Exception as e:
        print(f"\nError during pipeline test: {str(e)}")
        raise

if __name__ == "__main__":
    # Create a sample test.txt file if it doesn't exist
    # Modified test content in test_pipeline.py
    test_content = """This is a test document for our RAG system.
    It contains multiple sentences that will be processed through our pipeline.
    We will use this to verify our document loading, chunking, embedding, and retrieval capabilities."""
    
    # Ensure sample_corpus directory exists
    Path("sample_corpus").mkdir(exist_ok=True)
    
    # Write test content
    with open("sample_corpus/test.txt", "w") as f:
        f.write(test_content)
    
    # Run the test
    test_complete_pipeline()