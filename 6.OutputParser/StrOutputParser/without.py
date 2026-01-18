from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({'topic':'quantum computing'})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1.content})

result2 = model.invoke(prompt2)

print(result2.content)