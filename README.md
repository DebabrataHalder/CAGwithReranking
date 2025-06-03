# CAGwithReranking

📚 PDF Chat with Reranking + CAG
This application allows users to upload PDF documents and chat with them using advanced retrieval techniques powered by LangChain, Cohere, and Groq. It includes features like:

PDF text extraction

Chunking and embedding with Cohere

FAISS vector store

Reranking with Cohere Rerank

LLM-based compression using CAG (ChainExtractor)

Conversational QA via Groq’s LLaMA 3.3 70B

📁 Project Structure

project-root/
├── .env
├── requirements.txt
├── app.py
└── src/
    ├── __init__.py
    ├── document_processing.py
    ├── vector_store.py
    ├── retriever.py
    ├── chain_setup.py
    ├── ui_handlers.py
    └── utils.py
✅ Setup Instructions
1. Clone the Repository

git clone https://github.com/DebabrataHalder/pdfChatWithRerankingCAG.git
cd pdf-chat-rerank-cag
2. Install Dependencies
Make sure you have Python 3.10 or later installed.


pip install -r requirements.txt
3. Configure Environment Variables
Create a .env file in the root directory with the following:


COHERE_API_KEY=your_cohere_api_key
GROQ_API_KEY=your_groq_api_key
🔑 You can get your API keys from:
• Cohere Dashboard
• Groq Developer Portal

🚀 Running the App
To start the Streamlit app:


streamlit run app.py
The app will launch in your default browser at http://localhost:8501.
