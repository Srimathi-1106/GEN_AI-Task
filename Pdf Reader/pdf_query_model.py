import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os



# Initialize Gemini API

os.environ["GOOGLE_API_KEY"] = "AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34"

# Streamlit UI setup
st.title("Chat with your PDF using RAG")
st.sidebar.header("Upload PDF")

uploaded_file = st.sidebar.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name
    
    # Load and process PDF
    loader = PyPDFLoader(temp_pdf_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(pages)
    
    # Embedding and Vector DB setup
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    
    # Setup LangChain RAG with Gemini
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro")
    qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=retriever)
    
    st.session_state.qa_chain = qa_chain  # Store for later use
    st.success("PDF uploaded and indexed successfully!")

# Chat UI
def chat_interface():
    st.subheader("Chat with your PDF")
    user_query = st.text_input("Ask something from the PDF:")
    if st.button("Ask") and user_query:
        if "qa_chain" in st.session_state:
            response = st.session_state.qa_chain.run(user_query)
            st.write(response)
        else:
            st.warning("Please upload a PDF first.")

chat_interface()

# Cleanup temp file
if uploaded_file is not None:
    os.remove(temp_pdf_path)
