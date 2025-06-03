import os
import logging
import streamlit as st
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(text_chunks):
    """Create a FAISS vector store from text chunks using Cohere embeddings."""
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
