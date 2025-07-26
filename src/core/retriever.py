from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger

class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_documents",
        host: str = "localhost",
        port: int = 6333
    ):
        """Initialize Qdrant client and ensure collection exists"""
        try:
            self.client = QdrantClient(host=host, port=port)
            self.collection_name = collection_name
            self._create_collection_if_not_exists()
            logger.info(f"Connected to Qdrant collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {str(e)}")
            raise

    def _create_collection_if_not_exists(self, vector_size: int = 384):
        """Create collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,  # MiniLM-L6 dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Collection creation failed: {str(e)}")
            raise

    def store_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict]] = None
    ):
        """Store documents and their embeddings"""
        try:
            points = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                point = models.PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        "text": text,
                        **(metadata[i] if metadata else {})
                    }
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Stored {len(texts)} documents in vector store")
        except Exception as e:
            logger.error(f"Failed to store documents: {str(e)}")
            raise

    def retrieve(
        self,
        query_embedding: List[float],
        limit: int = 3
    ) -> List[Dict]:
        """Retrieve most similar documents"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            retrieved_docs = [
                {
                    "text": hit.payload["text"] if hit.payload and "text" in hit.payload else None,
                    "score": hit.score,
                    **({k: v for k, v in hit.payload.items() if k != "text"} if hit.payload else {})
                }
                for hit in results
            ]
            
            logger.debug(f"Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise