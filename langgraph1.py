import langgraph 
from typing import  TypedDict
from PIL import Image
import io
#from matplotlib.pylab import double 
class MyState(TypedDict):
    count: int
    
ms = MyState()
ms["count"] = 5
print(ms["count"]) 


def increment(st:MyState) -> MyState:
    return {"count": st["count"] + 1}
def double(st:MyState) -> MyState:
    return {"count": st["count"] * 2}


# eg to define a function with dtype
"""def abcd(cnt: int) -> int:
    return cnt + 10
print(abcd(5))"""

from langgraph.graph import StateGraph, END
graph = StateGraph(MyState)
graph.add_node("increment",increment)
graph.add_node("double",double)
graph.set_entry_point("increment")
graph.add_edge("increment", "double")
graph.add_edge("double",END)

app=graph.compile()
result=app.invoke({"count":10})
print(result)

image_data=app.get_graph().draw_mermaid_png()
img=Image.open(io.BytesIO(image_data))
img.show()