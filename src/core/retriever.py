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

    # In VectorStore.retrieve()
    def retrieve(self, query_embedding, limit=3, metadata_filter=None, score_threshold=0.3):
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit * 2,  # Fetch more initially
                with_payload=True
            )

            filtered = [hit for hit in results if hit.score >= score_threshold]
            if not filtered:
                logger.warning("No results above threshold, falling back to top results")
                filtered = results[:limit]


            seen, deduped = set(), []
            for hit in filtered:
                payload = hit.payload or {}  # Ensure payload is a dict
                text = payload.get("text", "").strip()
                if text and text not in seen:
                    deduped.append(hit)
                    seen.add(text)

            return [
                {
                    "text": (hit.payload or {}).get("text", ""),
                    "metadata": hit.payload or {},
                    "score": hit.score,
                }
                for hit in deduped[:limit]
            ]
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
