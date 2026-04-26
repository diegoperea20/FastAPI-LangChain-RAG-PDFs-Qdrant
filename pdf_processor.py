"""
PDF Processor Module
Handles PDF extraction and duplicate detection using file hash.

English:
- Each chunk includes a prefix with the filename as extra context,
  improving the embedding's semantics.

Spanish:
- Cada chunk lleva un prefijo con el nombre del archivo como contexto extra,
  mejorando la semántica del embedding.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
from langchain_core.documents import Document


class PDFProcessor:
    """Process PDF files and extract text with duplicate detection."""
    
    def __init__(self, pdf_directory: str):
        self.pdf_directory = Path(pdf_directory)
        self.chunk_size = 1000
        # ✅ MEJORA 1: overlap mayor para que secciones como "DATOS DE CONTACTO"
        # aparezcan en más chunks y no queden aisladas.
        self.chunk_overlap = 400
    
    def get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file for duplicate detection."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract all text from a PDF file using PyMuPDF."""
        text = ""
        try:
            doc = fitz.open(file_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return ""
        return text.strip()
    
    def split_text_into_chunks(self, text: str, source: str) -> List[Document]:
        """
        Split text into overlapping chunks for better retrieval.
        Each chunk includes a prefix with the filename.

        This gives the chunk embedding context about which company/document it belongs to,
        improving semantic retrieval when querying
        specific data from that company.
        Spanish:
        Cada chunk incluye un prefijo con el nombre del archivo.
        Esto hace que el embedding del chunk tenga contexto sobre qué empresa/documento
        pertenece, mejorando la recuperación semántica cuando se pregunta por
        datos específicos de esa empresa.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(text)
        documents = []
        
        # Nombre legible del documento (sin extensión) para el prefijo
        doc_name = Path(source).stem.replace("_", " ").replace("-", " ")
        
        for i, chunk in enumerate(chunks):
            # PREFIJO DE CONTEXTO: añade el nombre del documento al inicio
            # del chunk para que el embedding capture la identidad del documento.
            # Ejemplo: "Documento: empresa2 | Contenido:\n<texto del chunk>"
            enriched_content = f"Documento: {doc_name} | Contenido:\n{chunk}"
            
            doc = Document(
                page_content=enriched_content,
                metadata={
                    "source": source,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    # Guardamos el texto original sin prefijo por si se necesita
                    "raw_content": chunk,
                }
            )
            documents.append(doc)
        
        return documents
    
    def process_all_pdfs(self, existing_hashes: set = None) -> Tuple[List[Document], Dict[str, str]]:
        """
        Process all PDFs in the directory.
        Returns: (list of documents, dict of filename -> hash)
        """
        if existing_hashes is None:
            existing_hashes = set()
        
        all_documents = []
        file_hashes = {}
        new_files = []
        skipped_files = []
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        for pdf_file in pdf_files:
            file_hash = self.get_file_hash(pdf_file)
            file_hashes[pdf_file.name] = file_hash
            
            if file_hash in existing_hashes:
                skipped_files.append(pdf_file.name)
                print(f"Skipping {pdf_file.name} (already processed)")
                continue
            
            new_files.append(pdf_file.name)
            print(f"Processing {pdf_file.name}...")
            
            text = self.extract_text_from_pdf(pdf_file)
            if text:
                documents = self.split_text_into_chunks(text, pdf_file.name)
                for doc in documents:
                    doc.metadata["file_hash"] = file_hash
                all_documents.extend(documents)
                print(f"  Extracted {len(documents)} chunks from {pdf_file.name}")
        
        print(f"\nSummary:")
        print(f"  New files processed: {len(new_files)}")
        print(f"  Files skipped (already in DB): {len(skipped_files)}")
        print(f"  Total new chunks: {len(all_documents)}")
        
        return all_documents, file_hashes