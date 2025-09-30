"""Document processing module for loading and splitting documents"""

# Yahan par imports kiye gaye hain jo alag alag kaam karte hain:
# - WebBaseLoader: kisi bhi website URL se content (HTML/Text) load karne ke liye
# - PyPDFLoader: single PDF file load karne ke liye
# - TextLoader: text (.txt) file load karne ke liye
# - PyPDFDirectoryLoader: poori ek directory ke andar jitni PDF hain sab load karne ke liye
# - RecursiveCharacterTextSplitter: bada text chhote chunks mein todne ke liye (RAG ke liye zaroori hai)
# - Document: LangChain ka standard document object jo text + metadata hold karta hai

from typing import List, Union
from pathlib import Path
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    PyPDFDirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document


class DocumentProcessor:
    """Ye class documents ko load aur process karne ke liye banayi gayi hai"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Constructor function: jab tum class ka object banao ge to ye run hoga.
        
        Args:
            chunk_size: ek chunk mein kitne characters aayenge (default 500)
            chunk_overlap: har chunk ke beech mein kitni overlap hogi (default 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # ------------------ Loaders ------------------

    def load_from_url(self, url: str) -> List[Document]:
        """Ye method ek URL se document load karega"""
        loader = WebBaseLoader(url)
        return loader.load()
            
    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Ye method poori ek directory ke andar jitni PDF files hain un sab ko ek sath load karega"""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Ye method ek TXT file se document load karega"""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Ye method ek single PDF file load karega"""
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    
    # ------------------ Dispatcher ------------------

    def load_documents(self, sources: List[str]) -> List[Document]:
        """
        Ye ek generic method hai jo alag alag type ke sources handle karta hai.
        Tum isme URLs, PDF folder path, ya TXT file path pass kar sakte ho.
        
        Logic:
        - Agar source ek URL hai → load_from_url
        - Agar source ek directory hai → load_from_pdf_dir
        - Agar source ek .txt file hai → load_from_txt
        - Agar source ek .pdf file hai → load_from_pdf
        """
        docs: List[Document] = []
        for src in sources:
            path = Path(src)

            if src.startswith("http://") or src.startswith("https://"):
                docs.extend(self.load_from_url(src))

            elif path.is_dir():  
                docs.extend(self.load_from_pdf_dir(path))

            elif path.suffix.lower() == ".txt":
                docs.extend(self.load_from_txt(path))

            elif path.suffix.lower() == ".pdf":
                docs.extend(self.load_from_pdf(path))

            else:
                raise ValueError(
                    f"Unsupported source type: {src}. "
                    "Use URL, .txt file, .pdf file, or PDF directory."
                )
        return docs
    
    # ------------------ Splitter ------------------

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Ye method loaded documents ko chhote chunks mein todta hai.
        RAG ke liye zaroori hai kyunki agar pura PDF (100+ pages) ek sath bhej diya
        to LLM confuse hoga aur context window exceed ho jayegi.
        """
        return self.splitter.split_documents(documents)
    
    # ------------------ Pipeline ------------------

    def process_sources(self, sources: List[str]) -> List[Document]:
        """
        Ye ek pipeline method hai jo:
        - pehle documents load karega (load_documents se)
        - phir un documents ko chunks mein tod dega (split_documents se)
        
        Tum directly isko call karke ek end-to-end process kar sakte ho.
        """
        docs = self.load_documents(sources)
        return self.split_documents(docs)
