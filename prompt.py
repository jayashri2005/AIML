#prompt

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

prompt1 = PromptTemplate.from_template("Explain {topic} like I am {age} year.")
print(prompt1.format(topic="tars", age=5))

prompt3 = ChatPromptTemplate.from_template("Explain {topic} like I am {age} year.")

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful physics tutor."),
    ("human", "Explain {topic} to a {age} year old."),
])
#print(prompt2.format_messages(topic="gravity", age=5))    

