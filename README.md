# Retrieval-Augmented Generation (RAG) Prototype

This project is a **Retrieval-Augmented Generation (RAG) system** designed for **demonstration and assignment purposes**.  
It combines **document retrieval** (via embeddings stored in Qdrant) with **language generation** to answer user queries based on a given knowledge base.

---

## Features

- Loads and **chunks raw documents** into manageable text segments.
- Generates embeddings using [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`).
- Stores embeddings in a **Qdrant vector database** for similarity-based retrieval.
- Retrieves the most relevant document chunks for each query and generates a natural language answer.
- Provides **fallback responses** when no relevant context is found.
- Includes an **evaluation module** to measure performance (accuracy, speed, and relevance).

---

## Repository Setup Guide

Follow these steps to clone and run the project:

### 1. Clone the Repository

```bash
git clone https://github.com/Shubham6914/Rag-prototype.git
cd Rag-prototype

2. Set Up Python Environment

python -m venv rag_env
source rag_env/bin/activate      # On Linux/Mac
rag_env\Scripts\activate         # On Windows

# Then install dependencies:
pip install -r requirements.txt

3. Start Qdrant (Vector Database)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
Verify Qdrant is running:
docker ps

3. Project Structure

├── sample_corpus/           # Knowledge base documents (for demo/testing)
│   └── test_doc.txt         # Example technical documentation file
├── src/
│   ├── core/
│   │   ├── embeddings.py    # Embedding generation logic
│   │   ├── retriever.py     # Vector store (Qdrant) retrieval logic
│   │   └── agent.py         # RAG agent (retrieval + generation)
│   ├── utils/
│   │   ├── document_loader.py  # Loads text files
│   │   ├── chunking.py         # Splits documents into chunks
│   │   └── logging.py          # Logging utilities
├── run_demo.py               # Runs the RAG demo with the sample document
├── evaluate.py               # Runs evaluation on test queries
├── requirements.txt          # Dependencies
└── README.md                 # This documentation file
|__ main.py                   # main file for cli interface


Running the Demo

python run_demo.py

This will:
    Load sample_corpus/test_doc.txt.

    Chunk it into embeddings and store in Qdrant.

    Allow interactive querying via the console.

    Generate answers using the retrieved context.

Running Evaluation
    This will run a set of predefined queries and output a JSON file (or print results) including:

    Query text

    Generated answer

    Retrieved context chunks

    Processing time per query