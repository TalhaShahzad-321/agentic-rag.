"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

# Typing aur state imports
from typing import List, Optional
from src.state.rag_state import RAGState

# LangChain / LangGraph se imports
from langchain_core.documents import Document   # document object
from langchain_core.tools import Tool           # tools banane ke liye (retriever, wikipedia)
from langchain_core.messages import HumanMessage # user input ko agent tak bhejne ke liye
from langgraph.prebuilt import create_react_agent # prebuilt ReAct agent banane ke liye

# Wikipedia tool ke imports
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Ye class RAG workflow ke liye nodes rakhti hai"""

    def __init__(self, retriever, llm):
        # retriever = VectorStore ka retriever
        # llm = tumhara language model (Gemini ya koi aur)
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # ReAct agent ko lazy initialize karna (jab zaroorat ho tab banega)

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """
        Ye ek simple retriever node hai.
        - user ka sawaal leta hai (state.question se)
        - retriever ko invoke karta hai
        - jo documents aaye unko nayi state ke andar daal deta hai
        """
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )

    def _build_tools(self) -> List[Tool]:
        """
        Ye private method tools banata hai jo ReAct agent use karega.
        Tools do hain:
        1. retriever → tumhara apna indexed corpus se docs lana
        2. wikipedia → general knowledge ke liye Wikipedia search
        """

        # --- retriever tool function banaya ---
        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            
            # top 8 docs merge karte hain readable format mein
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        # retriever tool object
        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn,
        )

        # --- Wikipedia tool banate hain ---
        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """
        Ye private method ReAct agent banata hai tools ke sath.
        - retriever tool
        - wikipedia tool
        Aur ek system_prompt bhi dete hain jo agent ke behaviour ko guide karega.
        """
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        # ReAct agent banaya jo reasoning + action (tool calls) + final answer karega
        self._agent = create_react_agent(self.llm, tools=tools, prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Ye node final answer generate karti hai.
        Steps:
        - Agar agent pehli baar run ho raha hai to usko build karo
        - agent ko user ke sawaal ke sath invoke karo
        - result se final message nikaalo
        - RAGState return karo jisme question, retrieved_docs aur answer hoga
        """
        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]  # last message hamesha agent ka final jawab hota hai
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer."
        )
