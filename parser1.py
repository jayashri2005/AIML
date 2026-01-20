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

prompty = ChatPromptTemplate.from_template([
    ("system","Answer the query.\n{format_instructions}"),
    ("user","{query}")
])

prompt_with_instructions=prompty.partial(format_instructions=pydantic_parser.get_format_instructions())

chain = prompt_with_instructions | llmist | pydantic_parser

response = chain.invoke({"query":"Tell me a joke about parrots."})

print(response)