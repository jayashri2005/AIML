from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
load_dotenv()
import os

os.getenv("HF_TOKEN")

llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))




class KnowledgeGraph:
    def __init__(self):
        self.nodes={}
        self.edges={}

    def add_node(self, node_id, **attrs):
        self.nodes[node_id]=attrs
    
    def add_edge(self, src , relation, dst):
        self.edges.setdefault(src, []).append((relation,dst))

    def neighbors(self, node_id):
        return self.edges.get(node_id,[])

kg=KnowledgeGraph()
kg.add_node("Aspirin", type="drug")
kg.add_node("COX-1", type="enzyme")
kg.add_node("Prostaglandin", type="chemical")
kg.add_node("Inflammation", type="effect")

kg.add_edge("Aspirin", "inhibits", "COX-1")
kg.add_edge("COX-1", "produces", "Prostaglandin")
kg.add_edge("Prostaglandin", "causes", "Inflammation")

class GraphRAGState(TypedDict):
    query:str
    entities:list[str]
    knowledge_graph:KnowledgeGraph
    graph_facts:list[tuple[str,str,str]]
    context:str
    answer:str

def extract_entities(state: GraphRAGState) -> GraphRAGState:
    query = state["query"]
    kg = state["knowledge_graph"]
    entities = []
    for node in kg.nodes:
        if node.lower() in query.lower():
            entities.append(node)
    state["entities"] = entities
    return state

def traverse_graph(state: GraphRAGState) -> GraphRAGState:
    kg=state["knowledge_graph"]
    entities=state["entities"]

    facts=[]
    
    def dfs(node, depth, max_depth=3, visited=None):
        if visited is None:
            visited=set()

        if depth > max_depth or node in visited:
            return

        visited.add(node)

        for rel, nbr in kg.neighbors(node):
            facts.append((node ,rel , nbr))
            dfs(nbr , depth+1,max_depth ,visited)

    for e in entities:
        dfs(e,0)

    state["graph_facts"]=facts
    return state

def build_context(state:GraphRAGState)->GraphRAGState:
    lines=[
        f"- {s} {r} {o}"
        for s,r,o in state["graph_facts"]
    ]
    state["context"]="\n".join(lines)
    return state

def answer_llm(state: GraphRAGState) -> GraphRAGState:
    prompt=f"""
    Use only the following context to answer the question.
    Facts:
    {state['context']}
    Question: {state['query']}
    """
    state["answer"]=llm.invoke(prompt)
    return state
    

graph = StateGraph(GraphRAGState)
graph.add_node("extract_entities",extract_entities)
graph.add_node("traverse_graph",traverse_graph)
graph.add_node("build_context",build_context)
graph.add_node("answer_llm",answer_llm)

graph.set_entry_point("extract_entities")
graph.add_edge("extract_entities","traverse_graph")
graph.add_edge("traverse_graph","build_context")
graph.add_edge("build_context","answer_llm")

app=graph.compile()

initial_state: GraphRAGState={
    "query":"How does aspirin reduce inflammation?",
    "knowledge_graph":kg,
    "entities":[],
    "graph_facts":[],
    "context":"",
    "answer":""
}

result=app.invoke(initial_state)

print("\n--- GRAPH CONTEXT ---")
print(result["context"])

print("\n --- FINAL ANSWER PROMPT ---")
print(result["answer"])