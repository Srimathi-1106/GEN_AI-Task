import streamlit as st
import os
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set Google API Key (Replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34"

# Streamlit app title
st.title("📄 PDF Chatbot with Gemini")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    # Save the uploaded file
    # with open("uploaded.pdf", "wb") as f:
    #     f.write(uploaded_file.read())
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_pdf_path = temp_file.name

    # Load the PDF and store embeddings
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()
    vectorstore = FAISS.from_documents(docs, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))

    # Setup memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup the chat model (Using Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Create the chain
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory)

    # User input
    user_query = st.text_input("Ask a question about the PDF:")

    if user_query:
        response = qa_chain({"question": user_query})

        # Store in session state
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Bot", response["answer"]))

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")
