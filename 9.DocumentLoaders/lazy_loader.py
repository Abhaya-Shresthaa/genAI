# loaders used to load all the files in memory
# if file size is large high load in memory 
# so we use concept of lazy loading 
# loads on demand (generator of documents object) like did when learning cnn


from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path= 'directory_testing',
    glob= '*.pdf',      # import pdf files only
    loader_cls= PyPDFLoader
)

#one document is object is loaded once while needed
#when we use just load() are are loaded at once and then displayed (takes time)
# in lazy loading one doc object(i.e pdf page) is loaded then again deleted from memory 
# then another page is loaded then again deleted and so on
docs = loader.lazy_load()

# it prints total pages combined in all pdfs
# print(len(docs))

for document in docs:
    print(document.metadata)