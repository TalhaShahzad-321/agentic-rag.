"""Streamlit UI for Agentic RAG System - Simplified Version"""
# ↑ Ye sirf ek docstring hai jo batata hai ke ye file Streamlit ka user interface banane ke liye hai.
# Ye tumhara frontend hai jo backend RAG system ke upar chal raha hai.

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))
# ↑ Ye trick use hoti hai taake tumhare "src" folder ki files ko import kiya ja sake
# warna Python ko path nahi milega. Yani ab tum direct src ke modules use kar sakte ho.

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder
# ↑ Ye imports backend ke core modules ko lekar aa rahe hain jo tumne banaye hain.
# - Config → LLM aur chunking ka setup rakhta hai
# - DocumentProcessor → PDF/URL documents ko process karta hai
# - VectorStore → documents ka embedding aur retrieval handle karta hai
# - GraphBuilder → poora RAG graph banata hai (retriever + responder)

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)
# ↑ Ye Streamlit ka UI config hai (tab title, emoji icon, centered layout)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)
# ↑ Yahan thoda CSS diya gaya hai taake buttons ka style thoda custom ho jaye.

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []
# ↑ Ye function Streamlit ki session_state initialize karta hai.
# Streamlit apps har refresh pe reset ho jate hain, to session_state ek memory ke tarah hota hai
# jo variables ko ek session ke dauran save rakhta hai.
# - rag_system → backend graph store karega
# - initialized → flag rakhega ke system load hua ya nahi
# - history → pehle ki queries aur answers list ki tarah store hongi

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        # Process documents
        documents = doc_processor.process_urls(urls)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0
# ↑ Ye function tumhara poora backend system initialize karta hai.
# 1. Config se LLM load hota hai.
# 2. DocumentProcessor se documents URLs se process hote hain.
# 3. VectorStore create hota hai aur documents ka FAISS index banata hai.
# 4. GraphBuilder banake retriever aur LLM connect hota hai.
# st.cache_resource decorator ensure karta hai ke ye function bar-bar na chale,
# ek dafa run hone ke baad cached result mile.

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    # ↑ Jab app pehli dafa chalti hai to RAG system initialize hota hai aur user ko success msg milta hai.

    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    # ↑ Ye ek simple input form hai jahan user apna sawaal likhta hai aur submit karta hai.

    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                # ↑ Yahan RAG system ko user ka sawaal diya jata hai aur woh final state return karta hai
                # jisme answer + retrieved_docs dono hote hain.

                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                # Show retrieved docs in expander
                with st.expander("📄 Source Documents"):
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )
                # ↑ Is section main user ko answer dikhaya jata hai aur ek expander ke andar
                # retrieved documents ke short previews bhi dikhaye jate hain.

                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    # ↑ Is block main poora QnA cycle chalta hai.

    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")
    # ↑ Ye section last 3 searches ka chhota history log user ko dikhata hai.

if __name__ == "__main__":
    main()
# ↑ Agar file direct chalayi gayi ho (import nahi hui) to main() run hoga aur app start ho jayega.
