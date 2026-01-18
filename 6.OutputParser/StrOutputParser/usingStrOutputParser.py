from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task ="text-generation"
)

model = ChatHuggingFace(llm = llm)

#takes a topic 
template1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=['topic']
)
#output of template1 is input to this
template2 = PromptTemplate(
    template="Write a 4 line summary on the following text: {text}",
    input_variables=['text']
)

# instead of using result.content we can directly use this parser

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

#result = chain.invoke({'topic': 'quantum computing'}) ##more specific vayera dida
result = chain.invoke('quantum computing')

print(result)