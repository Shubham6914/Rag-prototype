# src/core/agent.py
from typing import Dict, List
from loguru import logger
from transformers.pipelines import pipeline
from src.core.embeddings import EmbeddingGenerator
from src.core.retriever import VectorStore
from src.utils.logging import log_retrieval_event

class RAGAgent:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_generator: EmbeddingGenerator,
        model_name: str = "google/flan-t5-large"  # Use a compatible text2text model

    ):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator

        try:
            logger.info("Initializing text generation model...")
            self.generator = pipeline(
                "text2text-generation",
                model=model_name,
                max_length=512,
                temperature=0.3,
                do_sample=False,
            )

            logger.info("Model initialized successfully.")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            raise

    # In RAGAgent._format_prompt()
    def _format_prompt(self, query: str, context: List[Dict]) -> str:
        context_text = "\n\n---\n\n".join(
            [f"Chunk {i+1}:\n{doc['text'].strip()}" for i, doc in enumerate(context)]
        )
        return (
            "You are a helpful assistant. Use ONLY the context below to answer.\n"
            "Combine all the context chunks into a single, detailed, step-by-step response.\n"
            "If the answer is not in the context, explicitly say: 'The information is not available in the provided context.'\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Step-by-step Answer:"
        )






    def process_query(self, query: str) -> Dict:
        try:
            query_embedding = self.embedding_generator.generate_embeddings([query])[0]
            context = self.vector_store.retrieve(
                query_embedding=query_embedding,
                limit=3 # or 5, depending on doc size
            )

            retrieved_docs = [doc['text'][:100] + "..." for doc in context]
            context_used = context[0]['text'] if context else "No context found"
            log_retrieval_event(query, retrieved_docs, context_used)

            if not context:
                logger.warning("No relevant context found.")
                return {"answer": "No relevant information found in documents."}

            prompt = self._format_prompt(query, context)
            response = self.generator(prompt)[0]['generated_text']

            logger.debug(f"Agent Generated Agent Response: {response.strip()}")

            return {
            "answer": response.strip(),
            "retrieved_chunks": [doc['text'] for doc in context],  # full context for debugging
            "context_used": context_used
        }

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"answer": "Error processing your query."}
