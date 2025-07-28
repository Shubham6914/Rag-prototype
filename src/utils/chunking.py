from typing import List

class TextChunker:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = words[i:i + self.chunk_size]
            chunks.append(" ".join(chunk))
            i += self.chunk_size - self.chunk_overlap
        return chunks
