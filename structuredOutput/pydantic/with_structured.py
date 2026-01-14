from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm )

class Review(BaseModel):
    key_themes: list[str] = Field(description="all key points here in a list")
    summary: str = Field(description='A brief summary of the review')
    sentiment: Literal['pos', 'neg'] = Field(description='return sentiment as positive or negative')
    reviewer:Optional[str] = Field(description='Reviewer of the given review')
    
   
structured_model = model.with_structured_output(Review)
 
result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")
   
                   
print()
print()
print(type(result))
print()
print(result.sentiment)
print()
#converting to dictionary
result_dict = dict(result)
print(result_dict['sentiment'])
