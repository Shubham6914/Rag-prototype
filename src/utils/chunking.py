from typing import List
from loguru import logger

class TextChunker:
    """Handles text chunking with consideration for memory constraints"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks with overlap
        Handles memory constraints for large documents
        """
        try:
            chunks = []
            start = 0
            text_length = len(text)

            while start < text_length:
                end = start + self.chunk_size
                
                # Handle the last chunk
                if end > text_length:
                    end = text_length
                
                # Find the last complete sentence in chunk if possible
                if end < text_length:
                    end = self._find_sentence_boundary(text, end)
                
                # Add chunk to list
                chunks.append(text[start:end])
                
                # Move start position considering overlap
                start = end - self.chunk_overlap

            logger.debug(f"Text chunked into {len(chunks)} segments")
            return chunks

        except Exception as e:
            logger.error(f"Error during text chunking: {str(e)}")
            raise

    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """Find the nearest sentence boundary after the given position"""
        sentence_endings = ['. ', '! ', '? ']
        min_end = position
        
        for ending in sentence_endings:
            end = text.find(ending, position - 30, position + 30)
            if end != -1 and end < min_end:
                min_end = end + 1
                
        return min_end
