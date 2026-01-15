from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task ="text-generation"
)

model = ChatHuggingFace(llm = llm)

class feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description= 'give the sentiment of the feedback')

parser = StrOutputParser()
parser1 = PydanticOutputParser(pydantic_object= feedback)


prompt1 = PromptTemplate(
    template= 'classify the sentiment of this feedback as positive or neagative: {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser1.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

classifierChain = prompt1 | model | parser1


#tuple of (condition, chain)
branchChain = RunnableBranch(
    (lambda x:x.sentiment == 'positive', prompt2 | model | parser),  #if
    (lambda x:x.sentiment == 'negative', prompt3 | model | parser),  #elif
    RunnableLambda(lambda x: "could not find sentiment")             #else
)

chain = classifierChain | branchChain

result = chain.invoke({'feedback': 'This is a worst phone'})

print(result)