import streamlit as st
import os
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import chromadb 
from dotenv import load_dotenv

load_dotenv()
# Set Google API Key (Replace with your actual key)
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Streamlit app title
st.title("PDF Chatbot with Gemini")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=f"./vector_db/storage")

# Create or connect to an existing Chroma collection
collection_name = "pdfs_collection"
collection = client.get_or_create_collection(collection_name)

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize list to store all documents
all_docs = []

# Process each uploaded PDF file
if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Load the PDF and extract content
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Extract text from the PDF and append to all_docs list
        all_docs.extend(docs)

        # Insert the extracted content into the Chroma collection (single vector database for all PDFs)
        pdf_content = "".join([doc.page_content for doc in docs])
        collection.add(
            documents=[pdf_content],
            metadatas=[{"filename": uploaded_file.name}],
            ids=[uploaded_file.name]
        )

    # Create a FAISS vectorstore for querying (use all_docs)
    vectorstore = FAISS.from_documents(all_docs, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

    # Setup memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup the chat model (Using Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Create the chain for answering questions
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)

    # User input
    user_query = st.text_input("Ask a question about the uploaded PDFs:")

    if user_query:
        response = qa_chain({"question": user_query})

        # Store in session state
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Bot", response["answer"]))

    # Display chat history
    st.subheader("Chats") 
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")

else:
    user_query = st.text_input("Ask a question about the uploaded PDFs:", disabled=True)

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")
    st.warning("Please upload one or more PDF files.")

