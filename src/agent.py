from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # 1. Retrieve top-k relevant chunks from the store.
        chunks = self.store.search(question, top_k)
        # 2. Build a prompt with the chunks as context.
        context = "\n".join([chunk["content"] for chunk in chunks])
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        # 3. Call the LLM to generate an answer.
        answer = self.llm_fn(prompt)
        return answer
