from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

#temperature defines how much deterministic or creative ans should be 
#max token is for limit the maximum output tokens to be
model = ChatOpenAI(model="gpt-4", temperature=0.6, max_completion_tokens=10)