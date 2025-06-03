import logging
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter

def extract_pdf_text(pdf_docs):
    """Extract text from a list of uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text
        except Exception as e:
            logging.error(f"Error processing {pdf.name}: {e}")
            st.warning(f"Error processing {pdf.name}: {e}")
    return text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split a long string into smaller chunks."""
    if not text.strip():
        return []
    
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return splitter.split_text(text)

