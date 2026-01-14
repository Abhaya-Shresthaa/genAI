from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from pydantic import BaseModel, Field
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task = "text-generation"
)
model = ChatHuggingFace(llm = llm )

json_schema = {
    "title": "Review",
    "type":"object",
    "properties": {
        "key_themes":{
            "type": "array",
            "items":{
                "type": "string"
            },
            "description": "all key points here in a list"
        },
        "summary":{
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment":{
            "type": "string",
            "enum": ["pos","neg"],
            "description":"return sentiment as positive or negative"
        },
        "reviewer":{
            "type":"string",
            "description":"Reviewer of the given review"
        }
    },
    "required": ["key_themes","summary" ,"sentiment"]    
}
   
structured_model = model.with_structured_output(json_schema)
 
result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")
   
                   
print()
print()
print(type(result))
print()
print(result['sentiment'])
print()
print()
print(result)
