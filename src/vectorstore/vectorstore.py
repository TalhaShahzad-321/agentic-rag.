# Initialize HuggingFace embeddings

from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---- Embedding Model Initialization ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

"""Vector store module for document embedding and retrieval"""
# ↑ Yeh module-level docstring hai jo batata hai
# ke yeh file "document embedding aur retrieval" ke liye use hoti hai.


# ---- VectorStore Class ----
class VectorStore:
    """Manages vector store operations"""
    # ↑ Yeh ek wrapper class hai jo FAISS aur embeddings ko manage karti hai.
    # Fayda: Har jagah FAISS ka direct code likhne ki zaroorat nahi,
    # ek hi interface se saari cheezein handle ho jaati hain.

    def __init__(self):
        """Initialize vector store with HuggingFace embeddings"""
        # ↑ Constructor jab class ka object banta hai.
        # Isme default embedding model already initialize kiya gaya hai.
        self.embedding = embeddings
        # ↑ Class ke andar ek fixed embedding model assign kiya gaya.
        # (currently all-MiniLM-L6-v2 use ho raha hai)

        self.vectorstore = None
        # ↑ Initially vectorstore None rakha hai, jab tak documents add na kiye jaayen.

        self.retriever = None
        # ↑ Retriever bhi None hai, yeh later initialize hoga jab FAISS create ho jaayega.


    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents
        
        Args:
            documents: List of documents to embed
        """
        # ↑ Yeh method documents ko leta hai, embeddings banata hai aur FAISS mein store karta hai.
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        # ↑ FAISS.from_documents automatic:
        # 1. Documents → embeddings mein convert karta hai
        # 2. FAISS index banata hai aur usme store karta hai

        self.retriever = self.vectorstore.as_retriever()
        # ↑ FAISS se ek retriever object banaya gaya hai
        # jo later queries ke against relevant documents retrieve karega.


    def get_retriever(self):
        """
        Get the retriever instance
        
        Returns:
            Retriever instance
        """
        # ↑ Agar retriever bana hi nahi (matlab create_vectorstore call hi nahi hua),
        # to error throw karega.
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever
        # ↑ Warna retriever object wapas kar dega.
        # Yeh retriever basically ek wrapper hai jo semantic search provide karta hai.


    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        # ↑ Yeh function ek query leta hai aur top-k documents return karta hai.
        # k ka default 4 hai → matlab query ke against top 4 similar docs aayenge.

        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        # ↑ Agar retriever bana hi nahi to error raise karega.

        return self.retriever.invoke(query)
        # ↑ Query ko retriever pe bhej deta hai aur results return kar deta hai.
        # 'invoke' internally FAISS similarity search chalata hai
        # aur jo documents sabse zyada similar hain unko wapas karta hai.
