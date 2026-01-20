#praser to deal with input like json,csv,xml etc and convert it to dictionary

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")


str_parser = StrOutputParser()
print(str_parser.parse("This is a simple string output."))

json_parser = JsonOutputParser()
json_data='{"name": "Alice", "age": 30, "city": "New York"}'
print(json_parser.parse(json_data))

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentence , formatted as a JSON object with single key 'explanation'.")

jsparser = JsonOutputParser()
formatted_prompt = prompt.format_messages(topic="quantum computing")
print(formatted_prompt)



prompt1 = ChatPromptTemplate.from_template(
    "Explain {topic} in one senteance."
)


llmist = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-120b"))
chain = prompt1 | llmist | str_parser
result = chain.invoke({"topic": "stars"})
print("Final Result:", result)

prompt2 = ChatPromptTemplate.from_template(
    "Explain {topic} in one sentence , formatted as a JSON object with single key 'explanation'."
)
chain2 = prompt2 | llmist | jsparser
result2 = chain2.invoke({"topic": "black holes"})
print("Final Result:", result2)


from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

class Answer(BaseModel):
    answer:str=Field(description="Direct answer to the question:")
    confidence:float=Field(description="Confidence level of the answer on a scale of 0 to 1")

pydantic_parser=PydanticOutputParser(pydantic_object=Answer)
pydantic_data='{"answer":"42","confidence":0.95}'

parsed_answer=pydantic_parser.parse(pydantic_data)
print(parsed_answer.answer,parsed_answer.confidence)  

prompty = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the query.\n{format_instructions}"),
        ("user", "{query}"),
    ]
)

prompt_with_instructions=prompty.partial(format_instructions=pydantic_parser.get_format_instructions())

chain = prompt_with_instructions | llmist | pydantic_parser

response = chain.invoke({"query":"Tell me a joke about parrots."})

print(response)