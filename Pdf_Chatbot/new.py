import streamlit as st
import os
import tempfile
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

# Set Google API Key (Replace with your actual key)
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Streamlit app title
st.title("PDF Chatbot with Gemini")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=f"./vector_db")

# Create or connect to an existing Chroma collection
collection_name = "pdfs_collection"
collection = client.get_or_create_collection(collection_name)

# Initialize the embedding model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Upload multiple PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

# Initialize session state for memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize session state for document embeddings
if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = {}

# Check if uploaded files are different from previous uploads
if uploaded_files:
    # Clear chat history if a new file is uploaded
    if "uploaded_files" in st.session_state and st.session_state.uploaded_files != [f.name for f in uploaded_files]:
        st.session_state.chat_history = []
        st.session_state.document_embeddings = {}  # Reset embeddings
        st.session_state.uploaded_files = [f.name for f in uploaded_files]  # Store the new uploaded files

    # Process the uploaded files
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.document_embeddings:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_pdf_path = temp_file.name

            # Load the PDF and extract content
            loader = PyPDFLoader(temp_pdf_path)
            docs = loader.load()

            # Initialize a text splitter for chunking (e.g., chunk by characters)
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)  # 500 chars per chunk, 50 chars overlap

            # Chunk the documents into smaller pieces
            chunked_docs = []
            for doc in docs:
                chunked_docs.extend(text_splitter.split_text(doc.page_content))  # Chunking the text

            # Generate embeddings (vectors) for the chunked content
            pdf_vectors = embedding_model.embed_documents(chunked_docs)

            # Store the vectors along with content and metadata in ChromaDB
            for i, chunk in enumerate(chunked_docs):
                collection.add(
                    documents=[chunk],
                    metadatas=[{"filename": uploaded_file.name, "chunk_index": i + 1}],
                    ids=[f"{uploaded_file.name}_chunk_{i + 1}"],
                    embeddings=[pdf_vectors[i]]  # Store the vector for this chunk
                )

            # Save the document embeddings to avoid reprocessing
            st.session_state.document_embeddings[uploaded_file.name] = pdf_vectors

    # Initialize Chroma vector store with the embedding model object itself
    vectorstore = Chroma(
        persist_directory="./vector_db/storage", 
        embedding_function=embedding_model  # Pass the embedding model object
    )

    # Setup memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup the chat model (Using Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Create the chain for answering questions using the Chroma retriever
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory
    )

    # Handle the user input
    user_query = st.text_input("Ask a question about the uploaded PDFs:")

    if user_query and (user_query != st.session_state.get("last_query", "")):
        # Perform the query using the chain
        response = qa_chain({"question": user_query})

        # Store in session state
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Bot", response["answer"]))

        # Remember the last query so that we don't run the same query on every rerender
        st.session_state.last_query = user_query

    # Display chat history
    st.subheader("Chats")
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")

else:
    # If no files are uploaded, initialize the vector store
    vectorstore = Chroma(
        persist_directory="./vector_db/storage", 
        embedding_function=embedding_model  # Pass the embedding model object
    )

    # Setup memory for chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup the chat model (Using Gemini 1.5 Flash)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    # Create the chain for answering questions using the Chroma retriever
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, 
        retriever=vectorstore.as_retriever(), 
        memory=memory
    )

    user_query = st.text_input("Ask a question about the uploaded PDFs:")

    if user_query and (user_query != st.session_state.get("last_query", "")):
        # Perform the query using the chain
        response = qa_chain({"question": user_query})

        # Store in session state
        st.session_state.chat_history.append(("User", user_query))
        st.session_state.chat_history.append(("Bot", response["answer"]))

        # Remember the last query so that we don't run the same query on every rerender
        st.session_state.last_query = user_query

    # Display chat history
    st.subheader("Chats")
    for role, text in st.session_state.chat_history:
        st.write(f"**{role}:** {text}")

    st.warning("Please upload one or more PDF files.")
