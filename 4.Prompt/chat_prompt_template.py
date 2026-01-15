from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# by this way it 
# doesnot works
# messages = [
#     SystemMessage(content='You are a helpful {domain} expert'),
#     HumanMessage(content='Explain in simple terms, what is {topic}')
# ]

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful {domain} expert'),
    ('human', 'Explain in simple terms, what is {topic}')
])

# NOTpromt = messages.invoke({'domain':'AI', 'topic':'Quantum Neural Networks'})
promt = chat_template.invoke({'domain':'AI', 'topic':'Quantum Neural Networks'})
# print(NOTpromt)
print(promt)