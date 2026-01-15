from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.1",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

#to know what was asked before and who sent that

messages = [
    SystemMessage(content='You are a helpful assistant'),
]


while True:
    user_input = input("You: ")
    if user_input == 'exit':
        break
    messages.append(HumanMessage(user_input))
    result = model.invoke(messages)
    messages.append(AIMessage(content=result.content))
    print(result.content)

print(messages)