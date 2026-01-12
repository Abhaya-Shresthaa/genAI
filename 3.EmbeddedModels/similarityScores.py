from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

docs = [
    "Donald Trump president of America",
    "Xii president of China",
    "Kathmandu is the capital of Nepal",
    "Paris is the capital of France"
]

query = "who is president of china?"

doc_embedding = embedding.embed_documents(docs)
query_embedding = embedding.embed_query(query)

print(cosine_similarity([query_embedding], doc_embedding))