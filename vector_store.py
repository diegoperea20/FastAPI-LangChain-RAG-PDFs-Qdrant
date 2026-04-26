"""
Vector Store Module
Modelo de embeddings
- ANTES: all-MiniLM-L6-v2  (solo inglés)
- AHORA: paraphrase-multilingual-MiniLM-L12-v2 (50+ idiomas, mismo tamaño 384 dims)
"""

import os
from typing import List, Dict, Optional, Set
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


class QdrantVectorStoreManager:
    COLLECTION_NAME = "pdf_documents"
    
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, google_api_key=None):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        # modelo multilingüe
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vector_store: Optional[QdrantVectorStore] = None
    
    def collection_exists(self):
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)
    
    def create_collection(self, vector_size=384):
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection: {self.COLLECTION_NAME}")
        else:
            print(f"Collection {self.COLLECTION_NAME} already exists")
    
    def get_existing_file_hashes(self) -> Set[str]:
        if not self.collection_exists():
            return set()
        existing_hashes = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME, limit=100,
                offset=offset, with_payload=True, with_vectors=False
            )
            for point in points:
                if point.payload and "file_hash" in point.payload:
                    existing_hashes.add(point.payload["file_hash"])
            if offset is None:
                break
        print(f"Found {len(existing_hashes)} already processed files in database")
        return existing_hashes
    
    def add_documents(self, documents: List[Document]) -> int:
        if not documents:
            return 0
        self.create_collection()
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            collection_name=self.COLLECTION_NAME,
            url="http://localhost:6333"
        )
        self.vector_store.add_documents(documents)
        print(f"Added {len(documents)} document chunks to vector store")
        return len(documents)
    
    def get_vector_store(self) -> QdrantVectorStore:
        if self.vector_store is None:
            self.create_collection()
            self.vector_store = QdrantVectorStore.from_existing_collection(
                embedding=self.embeddings,
                collection_name=self.COLLECTION_NAME,
                url="http://localhost:6333"
            )
        return self.vector_store
    
    def get_retriever(self, k=5):
        return self.get_vector_store().as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": 10, "lambda_mult": 0.7}
        )
    
    def similarity_search(self, query: str, k=6) -> List[Document]:
        return self.get_vector_store().similarity_search(query, k=k)
    
    def get_collection_info(self) -> Dict:
        if not self.collection_exists():
            return {"exists": False}
        info = self.client.get_collection(self.COLLECTION_NAME)
        return {"exists": True, "points_count": info.points_count, "status": info.status.value}
    
    def clear_collection(self):
        if self.collection_exists():
            self.client.delete_collection(self.COLLECTION_NAME)
            print(f"Deleted collection: {self.COLLECTION_NAME}")
            self.vector_store = None