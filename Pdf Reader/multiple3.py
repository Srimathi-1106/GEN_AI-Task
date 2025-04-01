import os
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import streamlit as st
from chromadb import Client
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

# Set Google API Key (Replace with your actual key)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAceKZACJb0Si9xZIVbwBL4dcS4veiAz34"

# Streamlit app title
st.title("📄 PDF Chatbot with Gemini")

# Define base directory, storage, and session directories
base_directory = "/pdf_reader"  # Root directory where the project lives
storage_directory = os.path.join(base_directory, "storage")
sessions_directory = os.path.join(storage_directory, "sessions")

# Ensure the storage and sessions directories exist
if not os.path.exists(sessions_directory):
    os.makedirs(sessions_directory)

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Define a custom system message (instruction) for the model
custom_prompt = "You are an intelligent assistant that provides precise and concise answers without referencing or mentioning 'based on the provided text'. Please provide clear responses to the user's questions."

# Initialize the Chroma client once for the entire session
# Set the global persist directory for all PDFs
global_persist_directory = os.path.join(sessions_directory, "vector_db")

# Create the vector_db folder if it doesn't exist
if not os.path.exists(global_persist_directory):
    os.makedirs(global_persist_directory)

# Initialize ChromaDB client with the global persist directory
client = Client(path=global_persist_directory)

# Create or get the collection that will store all the documents
collection_name = "all_pdfs_collection"
collection = client.get_or_create_collection(name=collection_name)

# Initialize FAISS vector store (it will be populated with all PDFs)
vectorstore = FAISS()

# Process each uploaded PDF file
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Create a session folder for each PDF (optional)
        session_folder = os.path.join(sessions_directory, os.path.splitext(uploaded_file.name)[0])  # Session folder named after PDF
        
        # Create the session folder if it doesn't exist
        if not os.path.exists(session_folder):
            os.makedirs(session_folder)

        # Load PDF and extract content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Load the PDF and extract content
        loader = PyPDFLoader(temp_pdf_path)
        docs = loader.load()

        # Extract text from the PDF for storage
        pdf_content = "".join([doc.page_content for doc in docs])

        # Insert the extracted content into the Chroma collection
        collection.add(
            documents=[pdf_content],
            metadatas=[{"filename": uploaded_file.name}],
            ids=[uploaded_file.name]
        )

        # Add the documents to the FAISS vector store (to allow retrieval across all PDFs)
        vectorstore.add_documents(docs)

    # Setup memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup the chat model (Using Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Set up the custom prompt using ChatPromptTemplate
    system_message = SystemMessagePromptTemplate.from_template(custom_prompt)
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    
    # Combine the system and human messages into the final prompt
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])

    # Create the conversational retrieval chain (using the vector store for all PDFs)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectorstore.as_retriever(), memory=memory, prompt=chat_prompt)

    # User input
    user_query = st.text_input("Ask a question about the PDFs:")

    if user_query:
        response = qa_chain({"question": user_query})

        # Store in session state
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Bot", response["answer"]))

    # Display chat history
    st.subheader("Chat History")
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")

else:
    st.warning("Please upload one or more PDF files.")
