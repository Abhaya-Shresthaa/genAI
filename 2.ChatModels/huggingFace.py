from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # this has a hosted endpoint
    task = "text-generation",
    max_new_tokens= 100,
    # provider="hf-inference",
)

model=ChatHuggingFace(llm=llm)
# model = ChatHuggingFace(llm = lmm)

result = model.invoke("what is deep learning")
print(result.content)