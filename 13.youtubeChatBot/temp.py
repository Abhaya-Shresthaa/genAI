import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

st.header('YouTube Transcript Chat')

# 1. Setup LLM
# Note: Ensure "openai/gpt-oss-20b" is a valid repo you have access to. 
# For testing, "HuggingFaceH4/zephyr-7b-beta" is a great alternative.
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# 2. Function to process video (Cached to prevent re-running on every click)
@st.cache_resource
def process_video(video_id):
    try:
        
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript = " ".join(snippet.text for snippet in transcript_list.snippets)
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.create_documents([transcript])
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# 3. UI Logic
vid_id = st.text_input("YouTube Video ID (e.g., dQw4w9WgXcQ)")

if st.button('Load Video'):
    if vid_id:
        with st.spinner("Analyzing transcript..."):
            vs = process_video(vid_id)
            if vs:
                st.session_state.video_loaded = True
                st.session_state.video_id = vid_id
                st.success("Video loaded! Ask away.")
    else:
        st.error("Please enter a Video ID")

# 4. Chat Logic
if st.session_state.get("video_loaded"):
    # Re-retrieve the vector store from cache
    vector_store = process_video(st.session_state.video_id)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        template="""Answer the question based ONLY on the following context:
        {context}
        
        Question: {question}
        Answer:""",
        input_variables=['context', 'question']
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Build the chain
    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | StrOutputParser()
    )

    user_question = st.text_input("Enter your question")
    if st.button("Ask"):
        if user_question:
            with st.spinner("Thinking..."):
                response = chain.invoke(user_question)
                st.markdown("### Answer:")
                st.write(response)
        else:
            st.warning("Please enter a question.")