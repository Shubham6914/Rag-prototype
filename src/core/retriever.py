from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from loguru import logger
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
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

    def retrieve(self, query_embedding, limit=3, metadata_filter=None):
        try:
            search_kwargs = {
                "collection_name": self.collection_name,
                "limit": limit,
            }

            # Convert metadata_filter to Qdrant Filter if provided
            if metadata_filter:
                search_kwargs["query_filter"] = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        ) for key, value in metadata_filter.items()
                    ]
                )

            results = self.client.search(
                query_vector=query_embedding,
                **search_kwargs
            )

            return [
                {
                    "text": hit.payload.get("text", "") if hit.payload else "",
                    "metadata": hit.payload if hit.payload else {},
                    "score": hit.score,
                }
                for hit in results
            ]

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []