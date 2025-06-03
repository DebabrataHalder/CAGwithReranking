

import os
import logging
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank, LLMChainExtractor

# Initialize environment and logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Core Processing Functions ---
def extract_pdf_text(pdf_docs):
    """Extract text from PDF files with error handling"""
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            text += "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logging.error(f"Error processing {pdf.name}: {e}")
            st.warning(f"Error processing {pdf.name}: {e}")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into manageable chunks"""
    if not text.strip():
        return []
    
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    """Create FAISS vector store with Cohere embeddings"""
    if not text_chunks:
        st.error("No text available for vectorization")
        return None

    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        st.error("Missing COHERE_API_KEY")
        return None

    try:
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0", 
            cohere_api_key=cohere_key
        )
        return FAISS.from_texts(text_chunks, embeddings)
    except Exception as e:
        logging.error(f"Vectorization error: {e}")
        st.error(f"Failed to create vector store: {e}")
        return None

def create_advanced_retriever(vectorstore, llm):
    """Create two-stage retriever with Cohere reranking and CAG compression"""
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        st.error("Missing COHERE_API_KEY for reranking")
        return None

    try:
        # Stage 1: Retrieve 20 initial documents
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
        
        # Stage 2: Cohere reranking to top 10
        reranker = CohereRerank(
            top_n=10,
            model="rerank-english-v3.0",
            cohere_api_key=cohere_key
        )
        reranked_retriever = ContextualCompressionRetriever(
            base_compressor=reranker,
            base_retriever=base_retriever
        )
        
        # Stage 3: LLM-based compression (CAG)
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=reranked_retriever
        )
    except Exception as e:
        logging.error(f"Retriever setup error: {e}")
        st.error(f"Failed to create advanced retriever: {e}")
        return None

def setup_conversation_chain(vectorstore):
    """Initialize the conversational QA system"""
    if not vectorstore:
        return None

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        st.error("Missing GROQ_API_KEY")
        return None

    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.5,
            api_key=groq_key
        )
        memory = ConversationBufferMemory(
            memory_key='chat_history', 
            return_messages=True
        )
        retriever = create_advanced_retriever(vectorstore, llm)
        
        if not retriever:
            return None
            
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            verbose=True
        )
    except Exception as e:
        logging.error(f"Conversation setup error: {e}")
        st.error(f"Failed to initialize conversation chain: {e}")
        return None

# --- UI Handlers ---
def handle_user_query(query):
    """Process user questions through the conversation chain"""
    if not st.session_state.conversation:
        st.warning("Process documents first")
        return

    try:
        response = st.session_state.conversation({'question': query})
        st.session_state.chat_history = response['chat_history']
        
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
    except Exception as e:
        logging.error(f"Query handling error: {e}")
        st.error(f"Error processing your question: {e}")

def process_documents(pdf_docs):
    """Full document processing pipeline"""
    with st.status("Processing documents...", expanded=True) as status:
        # Text extraction
        st.write("Extracting text from PDFs...")
        raw_text = extract_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.error("No text extracted. PDFs may be image-based")
            return False

        # Chunking
        st.write("Splitting text into chunks...")
        chunks = chunk_text(raw_text)
        if not chunks:
            st.error("Failed to create text chunks")
            return False

        # Vectorization
        st.write("Creating vector database...")
        vector_store = create_vector_store(chunks)
        if not vector_store:
            return False

        # Conversation setup
        st.write("Initializing AI chain with reranking + CAG...")
        conversation_chain = setup_conversation_chain(vector_store)
        if not conversation_chain:
            return False
            
        st.session_state.conversation = conversation_chain
        status.update(label="Processing complete!", state="complete")
        st.success("Documents processed successfully!")
        return True

# --- Main Application ---
def main():
    st.set_page_config(
        page_title="PDF Chat with Reranking & CAG", 
        page_icon=":books:",
        layout="centered"
    )
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Main interface
    st.title("📚 PDF Chat with Reranking + CAG")
    st.caption("Upload PDFs, then ask questions about their content")
    
    # Chat input
    if query := st.chat_input("Ask about your documents..."):
        handle_user_query(query)
    
    # Sidebar operations
    with st.sidebar:
        st.header("Configuration")
        
        # Document upload
        st.subheader("1. Upload Documents")
        pdf_docs = st.file_uploader(
            "Upload PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        # Processing trigger
        st.subheader("2. Process Documents")
        if st.button("Process Documents", use_container_width=True):
            if not pdf_docs:
                st.warning("Upload at least one PDF")
                return
            if not process_documents(pdf_docs):
                status.update(label="Processing failed", state="error")
        
        # Footer
        st.divider()
        st.caption("Built with LangChain, Cohere, and Groq")
        st.caption("Models: Cohere embed-english-v3.0 + rerank-english-v3.0, llama-3.3-70b-versatile")

if __name__ == '__main__':
    main()