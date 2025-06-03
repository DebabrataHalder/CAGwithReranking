import streamlit as st
from src.utils import load_env_and_init_logging
from src.ui_handlers import handle_user_query, process_documents

def main():
    # Load env variables and initialize logging
    load_env_and_init_logging()

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
                st.error("Document processing failed")

        # Footer
        st.divider()
        st.caption("Built with LangChain, Cohere, and Groq")
        st.caption("Models: Cohere embed-english-v3.0 + rerank-english-v3.0, llama-3.3-70b-versatile")

if __name__ == '__main__':
    main()
