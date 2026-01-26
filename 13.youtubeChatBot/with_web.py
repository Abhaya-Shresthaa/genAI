from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task ="text-generation",
    # temperature= 2
)




if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
    
st.header('Youtube Transcript')
model = ChatHuggingFace(llm = llm)

vid_id = st.text_input("Youtube Video ID")




if st.button('Load Video'):
    if not vid_id:
        st.error("Please enter a YouTube Video ID")
    else:
        st.session_state.video_loaded = True
        st.session_state.video_id = vid_id
     
     
     
if st.session_state.video_loaded:   
    video_id = st.session_state.video_id #id only
    try:
        #language na rakhe best language return garxa
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript = " ".join(snippet.text for snippet in transcript_list.snippets)
    except TranscriptsDisabled:
        st.error("No captions available for this video.")
        st.stop()
    splitter = RecursiveCharacterTextSplitter(chunk_size = 350, chunk_overlap = 90)
    chunks = splitter.create_documents([transcript])
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs={"k": 3})
    st.success("Video loaded successfully! Now ask your questions 👇")
    
    
    
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )
    question = "is trump wanting greenland"
    def format(retrieved_docs):
        context_text = "\n\n".join(docs.page_content for docs in retrieved_docs)
        return context_text
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format),
        'question': RunnablePassthrough()
    })
    parser = StrOutputParser()
    chain = parallel_chain | prompt | model | parser
    st.session_state.chain = chain


if st.session_state.video_loaded:
    user_question = st.text_input("Enter your question")

    if st.button("Ask"):
        result = st.session_state.chain.invoke(user_question)
        st.write(result)

