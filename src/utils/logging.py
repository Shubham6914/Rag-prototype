from loguru import logger
import sys
from typing import Dict, Any, List

# Configure basic logging
def setup_logging():
    """Configure basic logging setup with loguru"""
    # Remove default handler
    logger.remove()
    
    # Add custom handler with specific format
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO"
    )
    
    # Add file handler for complete reasoning traces
    logger.add(
        "logs/rag_traces.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="DEBUG",
        rotation="100 MB"
    )

def log_retrieval_event(query: str, retrieved_docs: List[str], context_used: str):
    """Log retrieval decisions and context used"""
    logger.debug(
        f"Query: {query}\n"
        f"Retrieved Documents: {retrieved_docs}\n"
        f"Context Used: {context_used}\n"
    )
