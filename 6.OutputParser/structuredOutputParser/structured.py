from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema 
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task ="text-generation",
)

model = ChatHuggingFace(llm = llm)


schema = [
    ResponseSchema(name = 'fact_1', description = "first fact about the topic"),
    ResponseSchema(name = 'fact_2', description = "second fact about the topic"),
    ResponseSchema(name = 'fact_3', description = "third fact about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 3 facts about the topic: {topic} \n {format_instructions}',
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt = template.invoke('quantum computing')

print(prompt)

LLMoutput = model.invoke(prompt)
result = parser.parse(LLMoutput)

print(result)