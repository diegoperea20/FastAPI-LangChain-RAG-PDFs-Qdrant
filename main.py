"""
RAG Chat Application with FastAPI, WebSocket, Gemini, and Qdrant.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import custom modules
from pdf_processor import PDFProcessor
from vector_store import QdrantVectorStoreManager
from rag_chain import RAGChain

# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# Configuration
PDF_DIRECTORY = Path(__file__).parent / "filesRAG"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat Assistant",
    description="AI Chat with PDF document retrieval using Gemini and Qdrant",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global instances
vector_store_manager: Optional[QdrantVectorStoreManager] = None
rag_chain: Optional[RAGChain] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global vector_store_manager, rag_chain
    
    print("\n" + "="*50)
    print("Initializing RAG System...")
    print("="*50)
    
    # Initialize vector store manager
    vector_store_manager = QdrantVectorStoreManager(
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Check Qdrant connection
    try:
        info = vector_store_manager.get_collection_info()
        print(f"Qdrant connection: OK")
        if info["exists"]:
            print(f"Collection 'pdf_documents': {info['points_count']} documents")
        else:
            print("Collection 'pdf_documents': Not found (will be created)")
    except Exception as e:
        print(f"ERROR: Cannot connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        print(f"Make sure Qdrant is running: docker run -d -p 6333:6333 qdrant/qdrant")
        raise e
    
    # Auto-process PDFs on startup
    print("\n" + "-"*50)
    print("Checking for PDFs to process...")
    print("-"*50)
    
    try:
        # Get existing file hashes from Qdrant
        existing_hashes = vector_store_manager.get_existing_file_hashes()
        
        # Initialize PDF processor
        pdf_processor = PDFProcessor(str(PDF_DIRECTORY))
        
        # Process PDFs (skip already processed)
        documents, file_hashes = pdf_processor.process_all_pdfs(existing_hashes)
        
        if documents:
            # Add new documents to vector store
            count = vector_store_manager.add_documents(documents)
            print(f"\n Processed {len(file_hashes)} PDFs")
            print(f" Added {count} document chunks")
            print(f" Files: {list(file_hashes.keys())}")
        else:
            print("No new PDFs to process (all files already in database)")
    except Exception as e:
        print(f"Warning: Could not process PDFs: {e}")
    
    # Initialize RAG chain
    rag_chain = RAGChain(
        vector_store_manager=vector_store_manager,
        google_api_key=GOOGLE_API_KEY
    )
    
    # Show final status
    info = vector_store_manager.get_collection_info()
    print("\n" + "="*50)
    print(f"RAG System ready with {info.get('points_count', 0)} documents")
    print("="*50 + "\n")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAG Chat Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "websocket": "/ws/chat",
            "process_pdfs": "/api/process-pdfs",
            "status": "/api/status"
        }
    }


@app.get("/api/status")
async def get_status():
    """Get system status."""
    try:
        info = vector_store_manager.get_collection_info()
        return {
            "status": "operational",
            "qdrant": {
                "connected": True,
                "collection_exists": info["exists"],
                "document_count": info.get("points_count", 0)
            },
            "pdf_directory": str(PDF_DIRECTORY),
            "pdf_files": list([f.name for f in PDF_DIRECTORY.glob("*.pdf")])
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.post("/api/process-pdfs")
async def process_pdfs():
    """Process all PDFs in the filesRAG directory."""
    try:
        # Get existing file hashes from Qdrant
        existing_hashes = vector_store_manager.get_existing_file_hashes()
        
        # Initialize PDF processor
        pdf_processor = PDFProcessor(str(PDF_DIRECTORY))
        
        # Process PDFs (skip already processed)
        documents, file_hashes = pdf_processor.process_all_pdfs(existing_hashes)
        
        if documents:
            # Add new documents to vector store
            count = vector_store_manager.add_documents(documents)
            return {
                "status": "success",
                "message": f"Processed {len(file_hashes)} PDFs, added {count} document chunks",
                "files_processed": list(file_hashes.keys()),
                "chunks_added": count
            }
        else:
            return {
                "status": "success",
                "message": "No new PDFs to process (all files already in database)",
                "files_processed": [],
                "chunks_added": 0
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/clear-database")
async def clear_database():
    """Clear the vector database (use with caution)."""
    try:
        vector_store_manager.clear_collection()
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for RAG chat."""
    await websocket.accept()
    print("WebSocket client connected")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                query = message_data.get("message", "")
                
                if not query:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Empty message"
                    })
                    continue
                
                # Send user message confirmation
                await websocket.send_json({
                    "type": "user_message",
                    "message": query
                })
                
                # Get relevant documents ONCE (for context display AND RAG)
                relevant_docs = rag_chain.get_relevant_documents(query, k=4)
                
                # Show only the MOST relevant source (first one with highest score)
                if relevant_docs and len(relevant_docs) > 0:
                    # Get the most relevant source
                    most_relevant_source = relevant_docs[0].metadata.get("source", "Unknown")
                    await websocket.send_json({
                        "type": "sources",
                        "sources": [most_relevant_source],
                        "context_preview": relevant_docs[0].page_content[:300] + "..."
                    })
                
                # Stream response from RAG chain - pass pre-retrieved docs
                await websocket.send_json({
                    "type": "stream_start",
                    "message": "Generating response..."
                })
                
                # Stream the response - SINGLE API CALL with pre-retrieved docs
                async for chunk in rag_chain.astream(query, docs=relevant_docs):
                    await websocket.send_json({
                        "type": "stream_chunk",
                        "chunk": chunk
                    })
                
                # Send completion message
                await websocket.send_json({
                    "type": "stream_complete",
                    "message": "Response complete"
                })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                print(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)}"
                })
    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    print("\nStarting RAG Chat Server...")
    print(f"PDF Directory: {PDF_DIRECTORY}")
    print(f"Qdrant: {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"WebSocket: ws://localhost:8000/ws/chat")
    print(f"API Docs: http://localhost:8000/docs\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )