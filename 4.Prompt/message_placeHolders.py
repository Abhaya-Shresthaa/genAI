from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


#chat template
chat_template = ChatPromptTemplate(
    [('system', 'You are a helpful AI expert'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')]
)
chat_history = []

# load message
with open('chat_history.txt') as f:
    chat_history = f.read().splitlines()

# print(chat_history)


#create prompt
prompt = chat_template.invoke({ 'chat_history': chat_history, 'query': "is that important"})

print(prompt)