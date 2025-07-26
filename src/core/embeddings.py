from sentence_transformers import SentenceTransformer
from typing import List
from loguru import logger

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding model"""
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Embedding model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(texts)
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise