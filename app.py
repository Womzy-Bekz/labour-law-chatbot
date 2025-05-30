import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Use Streamlit secrets for the OpenAI key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Load Labour Law content
@st.cache_resource
def load_qa_chain():
    loader = TextLoader("labour_law.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(texts, embeddings)

    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key, temperature=0),
        chain_type="stuff",
        retriever=retriever
    )
    return qa

qa_chain = load_qa_chain()

# Web interface
st.set_page_config(page_title="HR Labour Law Bot", page_icon="⚖️")
st.title("⚖️ Nigerian Labour Law Chatbot")
st.write("Ask me anything about employee rights, leave, termination, and more.")

user_question = st.text_input("Enter your question")

if user_question:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(user_question)
        st.success(answer)