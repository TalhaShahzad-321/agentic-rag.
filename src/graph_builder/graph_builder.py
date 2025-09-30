"""Graph builder for LangGraph workflow"""

# Import kar rahe hain LangGraph ka StateGraph aur END:
# - StateGraph: graph workflow banane ke liye use hota hai (har step ek node hota hai)
# - END: ek special node jo workflow ke khatam hone ko dikhata hai
from langgraph.graph import StateGraph, END

# RAGState: ye tumhari custom state class hai (src/state/rag_state.py mein hogi)
# jo graph ke har step mein state (question, docs, answer waghera) hold karti hai
from src.state.rag_state import RAGState

# RAGNodes: ye class tumne banayi hai (src/node/reactnode.py)
# isme actual functions (jaise retrieve_docs, generate_answer) define hain
from src.node.reactnode import RAGNodes


class GraphBuilder:
    """Ye class LangGraph ka workflow banati aur manage karti hai"""
    
    def __init__(self, retriever, llm):
        """
        Constructor function (jab object banta hai tab run hota hai)
        
        Args:
            retriever: VectorStore ka retriever (jo documents ko query ke hisaab se laata hai)
            llm: Language model instance (Gemini / OpenAI waghera)
        """
        self.nodes = RAGNodes(retriever, llm)  # nodes class ka object banaya jisme do functions hain
        self.graph = None   # initially graph empty hai
    
    def build(self):
        """
        Ye method poora RAG workflow graph build karta hai.
        
        Returns:
            Compiled graph instance
        """
        # Step 1: StateGraph banate hain jo RAGState ko follow karega
        builder = StateGraph(RAGState)
        
        # Step 2: Graph ke nodes add karte hain
        # "retriever" node → ye documents laane ke liye hoga (self.nodes.retrieve_docs)
        # "responder" node → ye final answer generate karega (self.nodes.generate_answer)
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        
        # Step 3: Entry point set karte hain (graph kis node se start hoga)
        builder.set_entry_point("retriever")
        
        # Step 4: Edges banate hain (kis node ke baad kaunsa node chalega)
        # retriever → responder
        # responder → END (matlab workflow yahin khatam ho jaye)
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        
        # Step 5: Graph ko compile kar dete hain (ab ye chalne ke liye ready hai)
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Ye method graph ko run karta hai ek input question ke sath.
        
        Args:
            question: User ka sawaal
        
        Returns:
            Final state (isme docs aur generated answer dono honge)
        """
        # Agar abhi tak graph build nahi hua, to build kar lo
        if self.graph is None:
            self.build()
        
        # Initial state banate hain user ke sawaal ke sath
        initial_state = RAGState(question=question)
        
        # Graph ko invoke karte hain (poora pipeline chalta hai)
        return self.graph.invoke(initial_state)
