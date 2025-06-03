import os
import logging
import streamlit as st
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank, LLMChainExtractor

def create_advanced_retriever(vectorstore, llm):
    """
    Build a two-stage retriever:
    1) Base retriever (k=20)
    2) Cohere reranker (top_n=10)
    3) LLM-based compressor (CAG)
    """
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

