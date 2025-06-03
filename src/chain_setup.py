import os
import logging
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from src.retriever import create_advanced_retriever

def setup_conversation_chain(vectorstore):
    """
    Initialize a ConversationalRetrievalChain using:
    - ChatGroq (Llama-3.3-70b-versatile)
    - Two-stage retriever (reranking + CAG compression)
    - Conversation memory buffer
    """
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

