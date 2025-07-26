from typing import List
from loguru import logger


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize with smaller default chunks for better performance"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        """Simple and efficient chunking implementation"""
        chunks = []
        if not text:
            return chunks

        # Simple sentence splitting
        sentences = text.replace('\n', ' ').split('. ')
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence = sentence.strip() + '. '
            sentence_size = len(sentence)

            if current_size + sentence_size > self.chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append(''.join(current_chunk))

        logger.debug(f"Created {len(chunks)} chunks from text")
        return chunks

    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """Find the nearest sentence boundary after the given position"""
        sentence_endings = ['. ', '! ', '? ']
        min_end = position
        
        for ending in sentence_endings:
            end = text.find(ending, position - 30, position + 30)
            if end != -1 and end < min_end:
                min_end = end + 1
                
        return min_end
