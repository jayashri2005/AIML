from dotenv import load_dotenv
load_dotenv()
from langgraph.prebuilt import create_react_agent
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from PIL import Image
import io 
llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))



def manager_node(state):
    task_input = state.get("task","")
    input = state.get("input","")
    prompt = f""" 
    You are a task manager. Based on the user request below, decide whether it is a;
    - translate
    - calculate
    -  summarize

    Respon with only one word (translate, summarize, calculate). 
    Task: {task_input}""" 
    decision = llm.invoke(prompt).content.strip().lower()
    return {"agent": decision, "input": input}

def translate_node(state):
    text = state.get("input","")
    prompt = f" Act like you are atranslater. Only respond with the English translation of the txt below:\n\n{text}"
    result=llm.invoke(prompt).content.strip()
    return {"result": result}
def summarize_node(state):
    text = state.get("input","")
    prompt = f" Summarize the following in 1-2 lines:\n\n{text}"
    result=llm.invoke(prompt).content.strip()
    return {"result": result}
def calculate_node(state):
    expression = state.get("input","")
    prompt = f" Please calculate and return the result of:\n\n{expression}"
    result=llm.invoke(prompt).content.strip()
    return {"result": result}

def route_by_agent(state):
    
    return{
    "translate": "translate",
    "summarize": "summarize",
    "calculate": "calculate",
    "input":state.get("agent","")
}.get(state.get("agent",""),"default")

def default_node(state):
    return {"result": "Sorry, I could not understand the task."}


from langgraph.graph import StateGraph

g=StateGraph(dict)

g.add_node("manager",manager_node)
g.add_node("translate",translate_node)
g.add_node("summarize",summarize_node)  
g.add_node("calculate",calculate_node)
g.add_node("default",default_node)

g.set_entry_point("manager")
g.add_conditional_edges("manager", route_by_agent)

g.set_finish_point("translate")
g.set_finish_point("summarize")
g.set_finish_point("calculate")
g.set_finish_point("default")

app = g.compile()

print(app.invoke({
    "task": "Can you translate this?",
    "input": "Bonjour le monde"
}))

print(app.invoke({
    "task": "Please summarize the following",
    "input": "LangGraph helps you build flexible multi-agent workflows in python.."
}))

respcal = app.invoke({
    "task": "what is 12 * 8 + 5?",
    "input": "12 * 8 + 5"
})

print(respcal['result'])

print(app.invoke({
    "task": "Can you dance?",
    "input": "foo"
}))


image_data=app.get_graph().draw_mermaid_png()
img=Image.open(io.BytesIO(image_data))
img.show()