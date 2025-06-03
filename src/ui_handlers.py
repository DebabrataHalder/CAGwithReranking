import logging
import streamlit as st
from src.document_processing import extract_pdf_text, chunk_text
from src.vector_store import create_vector_store
from src.chain_setup import setup_conversation_chain

def handle_user_query(query: str):
    """
    Send the user's question to the conversation chain and render the chat history.
    """
    if not st.session_state.conversation:
        st.warning("Process documents first")
        return

    try:
        response = st.session_state.conversation({'question': query})
        st.session_state.chat_history = response['chat_history']
        
        # Display each message in the chat history
        for i, message in enumerate(st.session_state.chat_history):
            role = "user" if i % 2 == 0 else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
    except Exception as e:
        logging.error(f"Query handling error: {e}")
        st.error(f"Error processing your question: {e}")

def process_documents(pdf_docs):
    """
    Full pipeline for:
    1. Extracting text
    2. Chunking text
    3. Creating vector store
    4. Setting up the conversation chain
    """
    with st.status("Processing documents...", expanded=True) as status:
        # 1. Text extraction
        st.write("Extracting text from PDFs...")
        raw_text = extract_pdf_text(pdf_docs)
        if not raw_text.strip():
            st.error("No text extracted. PDFs may be image-based")
            return False

        # 2. Chunking
        st.write("Splitting text into chunks...")
        chunks = chunk_text(raw_text)
        if not chunks:
            st.error("Failed to create text chunks")
            return False

        # 3. Vectorization
        st.write("Creating vector database...")
        vector_store = create_vector_store(chunks)
        if not vector_store:
            return False

        # 4. Conversation setup
        st.write("Initializing AI chain with reranking + CAG...")
        conversation_chain = setup_conversation_chain(vector_store)
        if not conversation_chain:
            return False
            
        st.session_state.conversation = conversation_chain
        status.update(label="Processing complete!", state="complete")
        st.success("Documents processed successfully!")
        return True
