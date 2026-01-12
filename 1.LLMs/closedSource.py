from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="GPT-4")
result = llm.invoke("How are you?")

print(result)