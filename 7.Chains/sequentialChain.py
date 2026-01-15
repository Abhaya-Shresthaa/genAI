from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task ="text-generation"
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(
    template= "Write a detailed report on {topic}",
    input_variables=['topic']
)
template2 = PromptTemplate(
    template="Write a 4 line summary on the following text: {text}",
    input_variables=['text']
)


parser = StrOutputParser()

chain_ho = template1 | model | parser | template2 | model | parser

#result = chain.invoke({'topic': 'quantum computing'}) ##more specific vayera dida
result = chain_ho.invoke('quantum computing')

print(result)

chain_ho.get_graph().print_ascii()