
# 🤖 Agentic RAG System with ReAct Agent

**A Retrieval-Augmented Generation (RAG) system using a ReAct agent for interactive document Q&A.**  
Process PDFs, TXT files, and URLs, retrieve relevant content, and generate precise answers with references in a Streamlit UI.

---

## 🎬 Demo
*Ask questions directly on your documents and get instant answers!*
---

## 🌟 Features

* Load documents from **PDFs, TXT files, and URLs**  
* Automatically **split documents** into RAG-friendly chunks  
* **Semantic search** with FAISS embeddings  
* **ReAct agent** for reasoning + tool usage (retriever + Wikipedia)  
* **Interactive Streamlit UI** for live Q&A  
* Maintain **query history** in session state  

---

## ⚡ Quick Start

### 1️⃣ Clone Repository

```bash
git clone https://github.com/TalhaShahzad-321/agentic-rag.git
cd Agentic-RAG-System
````

### 2️⃣ Setup Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Add Documents

* Place PDFs or TXT files in the `data/` folder.
* Optionally, add URLs in `data/urls.txt` (one URL per line).

### 4️⃣ Run Streamlit UI

```bash
streamlit run streamlit_app.py
```

### 5️⃣ Ask Questions

* Example:

```
summarize the "Attention Is All You Need" paper in 5 lines
```

### 6️⃣ Run Command-Line Mode (Optional)

```bash
python main.py
```

---

## 📁 Project Structure

```
Agentic-RAG-System/
│
├─ data/                      # PDFs, TXT files, URLs
├─ src/
│   ├─ config/                # LLM config, chunk size, default URLs
│   ├─ document_ingestion/    # DocumentProcessor class
│   ├─ vectorstore/           # VectorStore (FAISS) & retriever
│   ├─ graph_builder/         # RAG graph & agent orchestration
│   └─ node/                  # Nodes (ReAct agent)
├─ streamlit_app.py           # Streamlit UI frontend
├─ main.py                    # Interactive CLI entry point
├─ requirements.txt           # Dependencies
└─ README.md                  # Project documentation
```

---

## 🛠 How It Works

1. **Load Documents** → PDFs, TXT, URLs
2. **Chunk & Vectorize** → Split into smaller pieces & create embeddings
3. **Retrieve Relevant Chunks** → Using semantic search
4. **ReAct Agent** → Combines retrieved docs + external tools to reason and generate answers
5. **Deliver Answer** → Display answer with references

---

## 🔮 Future Enhancements

* Multi-language support for documents & external tools
* Integrate more sources (arXiv, GitHub, Wikipedia, etc.)
* Scalable vector store (Pinecone, Weaviate, etc.)
* Improved ReAct agent prompting and reasoning strategies

---

## 📚 References

* [LangChain Documentation](https://www.langchain.com/docs/)
* [LangGraph Docs](https://docs.langgraph.com/)
* [FAISS: Efficient Similarity Search](https://faiss.ai/)
* Vaswani et al., “Attention Is All You Need”, 2017

---

## 📝 License

MIT License © [Talha Shahzad]
