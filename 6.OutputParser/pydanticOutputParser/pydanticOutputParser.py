from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task ="text-generation",
    temperature= 2
)

model = ChatHuggingFace(llm = llm )


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt = 20, description="Age of the person")
    city: str = Field(description="Name of the city person lives")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me name age and city of a celebrity of {country} \n {format_instructions}", 
    input_variables=['country'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

#below steps in chain
# chain = template | model | parser
# final_result = chain.invoke('Nepal')


prompt = template.invoke({'country': 'Nepal'})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(prompt)
print()
print(result)
print()
print(result.content)
print()
print(final_result)