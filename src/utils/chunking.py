from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List

class TextChunker:
    def __init__(self, chunk_size=300, chunk_overlap=50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_text(self, text: str) -> List[str]:
        # Returns list of strings (not Document objects)
        docs = self.splitter.create_documents([text])
        return [doc.page_content for doc in docs]
