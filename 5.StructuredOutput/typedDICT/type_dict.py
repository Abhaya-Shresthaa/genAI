from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.1",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm )

#schema for dictionary to be
class Review(TypedDict):
    summary: str
    sentiment: str
   
structured_model = model.with_structured_output(Review)
 
result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print()
print()
print(type(result))
print()
print(result)
print()
print(result['sentiment'])
