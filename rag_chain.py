"""
RAG Chain Module
Implements Retrieval-Augmented Generation with Gemini and Qdrant.

MEJORAS:
- _setup_chain() ahora usa get_retriever() con MMR en lugar del retriever
  por defecto de similarity, para mayor diversidad de chunks recuperados.
- get_relevant_documents() también usa k=6 para red más amplia.
"""

import os
from typing import List, Dict, AsyncGenerator, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from vector_store import QdrantVectorStoreManager


RAG_SYSTEM_PROMPT = """Eres un asistente inteligente y útil que responde preguntas basándose en el contexto proporcionado.

Instrucciones:
1. Usa SOLAMENTE la información del contexto proporcionado para responder.
2. Si la información no está en el contexto, dilo claramente.
3. Sé conciso pero completo en tus respuestas.
4. Cita las fuentes cuando sea relevante (menciona de qué documento viene la información).
5. Si hay múltiples fuentes, sintetiza la información de manera coherente.

Contexto:
{context}

Historial de conversación:
{chat_history}
"""


class RAGChain:
    """RAG Chain with Gemini and Qdrant."""
    
    def __init__(
        self,
        vector_store_manager: QdrantVectorStoreManager,
        google_api_key: str = None,
        model_name: str = "gemini-2.5-flash"
    ):
        self.vector_store = vector_store_manager
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=google_api_key,
            temperature=0.7,
            streaming=True
        )
        self.chat_history: List[Dict] = []
        self._setup_chain()
    
    def _setup_chain(self):
        """
        Set up the RAG chain with retriever.
         usa get_retriever() con MMR para mayor diversidad de chunks.
        """
        # Usar MMR retriever en lugar del retriever por defecto
        self.retriever = self.vector_store.get_retriever(k=5)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into context string."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            formatted.append(f"[Documento {i} - {source}]\n{content}\n")
        return "\n".join(formatted)
    
    def _format_history(self) -> str:
        """Format chat history for context."""
        if not self.chat_history:
            return "Sin historial de conversación previo."
        
        formatted = []
        for msg in self.chat_history[-6:]:
            role = "Usuario" if msg["role"] == "user" else "Asistente"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)
    
    def add_to_history(self, role: str, content: str):
        """Add message to chat history."""
        self.chat_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
    
    async def astream(self, query: str, docs: List[Document] = None) -> AsyncGenerator[str, None]:
        """Stream response for a query. Optionally pass pre-retrieved documents."""
        self.add_to_history("user", query)
        
        if docs is None:
            docs = self.retriever.invoke(query)
        
        context = self._format_docs(docs)
        chat_history = self._format_history()
        
        system_message = RAG_SYSTEM_PROMPT.format(
            context=context,
            chat_history=chat_history
        )
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=query)
        ]
        
        full_response = ""
        try:
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    full_response += chunk.content
                    yield chunk.content
                else:
                    chunk_str = str(chunk)
                    full_response += chunk_str
                    yield chunk_str
        except Exception as e:
            yield f"\n[Error: {str(e)}]"
            return
        
        if full_response:
            self.add_to_history("assistant", full_response)
    
    async def query(self, query: str) -> str:
        """Get complete response for a query (non-streaming)."""
        full_response = ""
        async for chunk in self.astream(query):
            full_response += chunk
        return full_response
    
    def get_relevant_documents(self, query: str, k: int = 6) -> List[Document]:
        """
        Get relevant documents for a query without generating response.
        """
        return self.vector_store.similarity_search(query, k=k)