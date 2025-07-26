from pathlib import Path
from typing import List, Union, Dict
import markdown
from pypdf import PdfReader
from loguru import logger

class DocumentLoader:
    """Handles loading and basic processing of documents"""
    
    @staticmethod
    def load_document(file_path: Union[str, Path]) -> str:
        """
        Load document content based on file extension
        Supports: .txt, .md, .pdf
        """
        file_path = Path(file_path)
        
        try:
            if file_path.suffix == '.txt':
                return DocumentLoader._load_text(file_path)
            elif file_path.suffix == '.md':
                return DocumentLoader._load_markdown(file_path)
            elif file_path.suffix == '.pdf':
                return DocumentLoader._load_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise

    @staticmethod
    def _load_text(file_path: Path) -> str:
        """Load content from text file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def _load_markdown(file_path: Path) -> str:
        """Load and convert markdown to text"""
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()
            return markdown.markdown(md_content)

    @staticmethod
    def _load_pdf(file_path: Path) -> str:
        """Load content from PDF file"""
        reader = PdfReader(str(file_path))
        return " ".join(page.extract_text() for page in reader.pages)
