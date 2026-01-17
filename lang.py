from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel
import os

from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")
llm=ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))
# prompt=ChatPromptTemplate.from_template("Explain {topic} in one sentence,formatted as a JSON Object with a single key name explanation")
# jsparser=JsonOutputParser()
# chain=prompt|llm|jsparser
# result=chain.invoke({'topic':'LCEL in Lagchain'})
# print(result)
# prompt2=ChatPromptTemplate.from_template("Explain {topic} in one sentence.")
# def clean_topic(x):
#     return {"topic":x["topic"].strip().lower()}
# preprocess=RunnableLambda(clean_topic)
parser=StrOutputParser()
# chain2=preprocess|prompt2|llm|parser
# chain2.invoke({'topic':'LangChainLCEL'})
def validate(x):
    if "?" in x:
        return "good"
    return "bad"
validator=RunnableLambda(validate)
fix_prompt=ChatPromptTemplate.from_template("Fix this sentence to be a question: {sentence}")
chat_prompt=ChatPromptTemplate.from_template("Answer this question: {question}")
chain=(chat_prompt|llm|parser|validator|{
    "good":RunnableLambda(lambda x: f"Final Answer: {x}"),
    "bad":fix_prompt|llm|parser
})
result2=chain.invoke({'question':'What is LangChain?'})
print(result2)