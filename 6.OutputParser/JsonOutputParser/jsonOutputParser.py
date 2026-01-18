from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task ="text-generation",
    temperature= 0
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template= "Give me name, age, height, place of a celebrity \n {format_instructions}",
    input_variables=[{}],
    partial_variables={'format_instructions': parser.get_format_instructions()}    
)

#all 3 steps below just in a single using chain
# chain = template | model | parser
# result = chain.invoke({})

prompt = template.format()
print()
print(prompt)

tempResult = model.invoke(prompt)
result = parser.parse(tempResult.content)
print()
print(result)
print()
print(type(result))

