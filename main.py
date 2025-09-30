"""Main application entry point for Agentic RAG system"""
# ↑ Ye docstring sirf ye batata hai ke ye file tumhara system start karne ka entry point hai.


import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))
# ↑ Ye line ensure karti hai ke Python ko tumhara "src" folder ka path mil jaye,
# warna woh modules import nahi kar paata.


# Ab tumhare backend ke core modules import kiye ja rahe hain
from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder


class AgenticRAG:
    """Main Agentic RAG application"""
    
    def __init__(self, urls=None):
        """
        Initialize Agentic RAG system
        
        Args:
            urls: List of URLs to process (uses defaults if None)
        """
        print("🚀 Initializing Agentic RAG System...")
        
        # Agar user ne custom URLs nahi diye to default wale use kar lo
        self.urls = urls or Config.DEFAULT_URLS
        
        # LLM ko load karo (Config se)
        self.llm = Config.get_llm()
        
        # Document processor banate hain jo docs ko chunks mein split karega
        self.doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # VectorStore create karne ke liye object
        self.vector_store = VectorStore()
        
        # Pehle documents load aur process karna
        self._setup_vectorstore()
        
        # GraphBuilder banake retriever aur llm ko jodna
        self.graph_builder = GraphBuilder(
            retriever=self.vector_store.get_retriever(),
            llm=self.llm
        )
        self.graph_builder.build()
        
        print("✅ System initialized successfully!\n")
    
    def _setup_vectorstore(self):
        """Setup vector store with processed documents"""
        print(f"📄 Processing {len(self.urls)} URLs...")
        documents = self.doc_processor.process_urls(self.urls)
        print(f"📊 Created {len(documents)} document chunks")
        
        print("🔍 Creating vector store...")
        self.vector_store.create_vectorstore(documents)
    
    def ask(self, question: str) -> str:
        """
        Ask a question to the RAG system
        
        Args:
            question: User question
            
        Returns:
            Generated answer
        """
        print(f"❓ Question: {question}\n")
        print("🤔 Processing...")
        
        # Graph run karo (retriever + agent + llm)
        result = self.graph_builder.run(question)
        answer = result['answer']
        
        print(f"✅ Answer: {answer}\n")
        return answer
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print("💬 Interactive Mode - Type 'quit' to exit\n")
        
        # Yeh loop tab tak chalega jab tak user quit na likhe
        while True:
            question = input("Enter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question:
                self.ask(question)
                print("-" * 80 + "\n")


def main():
    """Main function"""
    # Dekhna ke agar data/urls.txt file exist karti hai to wahan se URLs le lo
    urls_file = Path("data/urls.txt")
    urls = None
    
    if urls_file.exists():
        with open(urls_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    
    # System initialize karna
    rag = AgenticRAG(urls=urls)
    
    # Example questions run karna
    example_questions = [
        "What is the concept of agent loop in autonomous agents?",
        "What are the key components of LLM-powered agents?",
        "Explain the concept of diffusion models for video generation."
    ]
    
    print("=" * 80)
    print("📝 Running example questions:")
    print("=" * 80 + "\n")
    
    for question in example_questions:
        rag.ask(question)
        print("=" * 80 + "\n")
    
    # User se puchhna ke kya woh interactive mode chalana chahta hai
    print("\n" + "=" * 80)
    user_input = input("Would you like to enter interactive mode? (y/n): ")
    if user_input.lower() == 'y':
        rag.interactive_mode()


if __name__ == "__main__":
    main()
# ↑ Agar ye file directly run ho rahi ho to "main()" function call ho jata hai
