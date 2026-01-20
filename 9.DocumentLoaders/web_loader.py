from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/Forest"

loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)
