from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "Donald Trump president of America",
    "Xii president of China",
    "Kathmandu is the capital of Nepal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(docs)
print(str(vector))