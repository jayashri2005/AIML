import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_node("Aspirin", type="Drug")
G.add_node("Inflammation", type="Disease")
G.add_edge("Aspirin", "Inflammation", relation="treats")

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, arrowsize=20)
edge_labels = nx.get_edge_attributes(G, "relation")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.savefig('graph.png')  
plt.close() 

"""
from graphviz import Digraph
dot=Digraph()
dot.node("A","Aspirin\n(Drug)")
dot.node("B","Inflammation\n(Disease)")
dot.edge("A","B",label=" treats ")
print(dot)



from langgraph.graph import StateGraph, END

class State(dict):
    pass
def retrieve(state):
    return {"step":"retrieve"}
def rerank(state):
    return {"step":"rerank"}
def answer(state):
    return {"step":"answer"}
graph=StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("rerank", rerank)
graph.add_node("answer", answer)

"""