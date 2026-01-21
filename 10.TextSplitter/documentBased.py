from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language= Language.PYTHON, ## MARKDOWN for markdown characters
    chunk_size = 350,
    chunk_overlap = 0
)

text = """class llm(Runnables):
    def __init__(self):
        openAIModel = HuggingFaceEndpoint(
            repo_id="openai/gpt-oss-20b",
            task ="text-generation"
        )
        self.model = ChatHuggingFace(llm = openAIModel)
        print("llm created")
        
    def invoke(self, prompt):
        return self.model.invoke(prompt)
class PromptTemplate(Runnables):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables
        
    def invoke(self, input_dict):
        return self.template.format(**input_dict)
        
template = PromptTemplate(
    template='Write a poem on hello',
    input_variables=['length', 'topic']
)
model = llm()"""

chunks = splitter.split_text(text)


for chu in chunks:
    print(chu)
    print()
    print()
    print()
    