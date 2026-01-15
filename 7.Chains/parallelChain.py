from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task ="text-generation"
)

model1 = ChatHuggingFace(llm = llm1)

llm2 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task ="text-generation"
)

model2 = ChatHuggingFace(llm = llm2)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate summarized note of the following text: \n {text}",
    input_variables=['text'],
)

prompt2 = PromptTemplate(
    template="Generate 3 quiz questions on the topic: \n {topic}",
    input_variables=['topic']
)

prompt3 = PromptTemplate(
    template= "There is summary of a topic and quiz question of a topic just join them, no need that they are of same topic  \n Summary1: {summary} \n quizQuestions: {questions}",
    input_variables=['summary', 'questions']
)
text = """
What is BLEU?
The acronym BLEU refers to a “Bilingual Evaluation Understudy,” and it’s a statistic for measuring the accuracy of machine translations compared to human translators. IBM’s version of BLEU NLP is a popular tool for analyzing data and gauging the quality of machine translations.
* The machine translation is compared to one or more reference translations through BLEU by comparing n-grams (word sequences).
The machine-generated translation is given a score between 0 and 1 based on the measure, with 1 signifying full agreement with the reference translation. A machine translation’s accuracy is measured by how well it matches the source text’s n-gram frequency distribution.
When comparing translations across languages with different grammatical structures or word order, the BLEU metric might provide deceptive findings due to its limitations. Yet, because of its simplicity and convenience, it continues to be one of the most used measures for evaluating machine translation.
"""

parallelChain = RunnableParallel({
    'summary': prompt1 | model1 | parser,
    'questions': prompt2 | model2 | parser
})

afterParallel = prompt3 | model1 | parser

combinedChain = parallelChain | afterParallel
result = combinedChain.invoke({'text':text, 'topic': "Quantum computing"})

# tempResult1 = parser.parse(model1.invoke(prompt1.invoke({'text': text})))

# tempResult2 = parser.parse(model2.invoke(prompt2.invoke({'topic': "Quantum computing"})))
# result = parser.parse(model2.invoke(prompt3.invoke({'summary': tempResult1, 'questions': tempResult2})))



print(result)

combinedChain.get_graph().print_ascii()


##checking working of model2 and questions
# tempResult = parser.parse(model2.invoke(prompt2.invoke({'topic': "Quantum computing"})))

# print(tempResult)