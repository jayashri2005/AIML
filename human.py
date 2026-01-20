import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableWithMessageHistory

llmhg=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))
"""mess=[]
mess.append(SystemMessage(content="Please refine the answer according to exam"))
mess.append(HumanMessage(content="what is spoken in france?"))
mess.append(AIMessage(content="french haha hehe"))
mess.append(HumanMessage(content="what is spoken in spanish?"))
mess.append(AIMessage(content="spanish haha hehe"))
mess.append(HumanMessage(content="what is spoken in india?"))

mess=[
    SystemMessage(content="Please refine the answer according to exam"),
    HumanMessage(content="what is spoken in france?"),
    AIMessage(content="french haha hehe"),
    HumanMessage(content="what is spoken in spanish?"),
    AIMessage(content="spanish haha hehe"),
    HumanMessage(content="what is spoken in india?"),
]


response1=llmhg.invoke(mess[-1].content)
print("Response from HuggingFace",response2.content)


response2=llmhg.invoke(mess)
print("Response from hugging face llm",response2.content)



meme=InMemoryChatMessageHistory()
meme.add_message(SystemMessage(content="Please refine the answer according to example given"))
meme.add_user_message("what is spoken in france?")
meme.add_ai_message("french haha hehe")
meme.add_user_message("what is spoken in spain?")
meme.add_ai_message("spain haha hehe")
meme.add_user_message("what is spoken in italy?")
                                                                                                                       
#print(meme)
reponse3=llmhg.invoke(meme.messages)
print("Response from HuggingFace",reponse3.content)

"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "Please refine you answer according to example given"),
    MessagesPlaceholder(variable_name="history"),
    #("human","{user_input}")
    #("human", "what is spoken in Russia?")
    ("human", "{input}")
    ])

store={}
def get_history(session_id: str):  #key: session identifier
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
"""
get_history("harry").add_message("what is spoken in germany?")
get_history("harry").add_message("what is capital of germany?")
get_history("harry").add_message("what is spoken in portugal?")     
get_history("david").add_message("what is spoken in brazil?")

#print(store)
print(prompt.format_prompt(history=get_history("harry").messages))   #to extract the messages from history
#print(prompt.messages)
"""


app=RunnableWithMessageHistory(
    runnable= prompt | llmhg,
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history",
)
response4= app.invoke(
    {"input": "what is spoken in germany?"},
    config={"configurable": {"session_id": "harry"}}
)

response5 = app.invoke(
    {"input": "what is spoken in germany?"},
    config={"configurable": {"session_id": "harry"}}
)

print(response4.content)
print(response5.content)