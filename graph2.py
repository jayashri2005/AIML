from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):  # Proper state schema
    step: str
    documents: list  # For RAG docs
    query: str

def retrieve(state: State) -> dict:
    print("Retrieving documents...")
    return {"step": "retrieve", "documents": ["doc1 on inflammation", "doc2 on aspirin"]}  # Mock retriever

def rerank(state: State) -> dict:
    print("Reranking...")
    return {"step": "rerank", "documents": state["documents"][:3]}  # Mock top-3

def answer(state: State) -> dict:
    print("Generating answer...")
    return {"step": "answer"}

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("rerank", rerank)
graph.add_node("answer", answer)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "answer")
graph.add_edge("answer", END)

app = graph.compile()

# Run with initial state
result = app.invoke({"step": "start", "query": "Aspirin for inflammation?"})
print(result)
