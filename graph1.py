import os
graphviz_bin = r'C:\Graphviz-14.1.1-win64\bin'  # Your extracted path
os.environ['PATH'] = graphviz_bin + os.pathsep + os.environ['PATH']

from graphviz import Digraph
dot = Digraph()
dot.node("A", "Aspirin\n(Drug)")
dot.node("B", "Inflammation\n(Disease)")
dot.edge("A", "B", label="treats")
print(dot.source)  # Prints DOT source; render with dot -Tpng graph.dot -o graph.png
dot.render('graph', format='png', view=False)  # Auto-saves graph.png
