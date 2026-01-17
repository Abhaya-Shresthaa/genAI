from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task = 'text-generation'
)

model1 = ChatHuggingFace(
    llm = llm1
)

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task = 'text-generation'
)

model2 = ChatHuggingFace(
    llm = llm2
)


def word_count(text):
    return len(text.split())

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'just generate a joke on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="explain the joke: {joke}",
    input_variables=['joke']
)

prompt3 = PromptTemplate(
    template="summarize this joke in less words {joke}",
    input_variables=['joke']
)



first_chain = RunnableSequence(prompt1, model1,parser )

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(), #using pass through to let it pass without change
    'explain' : RunnableSequence(prompt2,model2, parser ),
    'words_count': RunnableLambda(word_count)
})
# lambda x: {
#     **x,
#     'joke': (
#         RunnablePassthrough()
#         | prompt3
#         | model1
#         | parser
#     ).invoke(x['joke'])
# }

branched_chain = RunnableBranch(
    (lambda x: x['words_count'] > 20, RunnableLambda(lambda x :{**x, 'joke': RunnableSequence(prompt3, model1,parser).invoke({'joke': x['joke']})})),
    RunnablePassthrough()
)

final_chain = first_chain | parallel_chain | branched_chain

result = final_chain.invoke({'topic': 'airplane'})

print(result)

# final_chain.get_graph().print_ascii()