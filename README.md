# RAG-Based Product Recommendation Chatbot (Flipkart)

A Retrieval-Augmented Generation (RAG) chatbot that recommends products based on user queries, built using LangChain, AstraDB, and Groq LLM.

## Overview
This project implements a conversational product recommendation system for an e-commerce use case (Flipkart-style). It uses RAG architecture to retrieve relevant product information from a vector database and generate context-aware responses using a fast LLM (Groq).

## Tech Stack
- Python
- Flask (web application)
- LangChain (RAG orchestration)
- AstraDB (vector database)
- Groq LLM (fast inference)
- HuggingFace Sentence Transformers (local embeddings)

## Dataset
- Flipkart product review dataset, used to build the knowledge base for retrieval

## Architecture & Approach
1. **Data Ingestion** — product review data is converted into structured documents and embedded using local HuggingFace sentence-transformer embeddings
2. **Vector Storage** — embeddings are stored in AstraDB for fast similarity search
3. **Retrieval** — given a user query, relevant product documents are retrieved from AstraDB based on vector similarity
4. **Generation** — retrieved context is passed to Groq LLM via LangChain to generate a natural language recommendation response
5. **Intent Detection** — a query classification system (covering 12 intent classes) routes queries appropriately before retrieval
6. **Web Interface** — Flask-based chatbot interface for user interaction

## Key Engineering Challenges Solved
- Migrated from LangChain 0.x to 1.x API by rewriting chains using `RunnableLambda`
- Handled deprecated Groq model versions
- Fixed AstraDB collection management issues
- Resolved data ingestion loop bugs
- Managed Groq API TPM (tokens-per-minute) rate limiting

## Project Structure
```
├── MODULES/
│   ├── data_converter.py       # Converts raw data into document format for embedding
│   └── retrieval_generation.py # RAG retrieval and generation logic
├── static/                     # Static assets (CSS/JS)
├── templates/                  # HTML templates for Flask frontend
├── app.py                      # Flask application entry point
├── astradata_upload.py         # Script to upload/index data into AstraDB
├── setup.py                    # Package setup
├── template.py                 # Project structure template generator
├── flipkart_product_review.xlsx # Source dataset
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
python -m venv venv
venv\Scripts\activate       # On Windows
pip install -r requirements.txt
python astradata_upload.py    # Upload data to AstraDB (run once)
python app.py                 # Start the Flask app
```

**Note**: 
- Requires a `.env` file with AstraDB and Groq API credentials (not included in this repo for security).
- The `venv/` folder (virtual environment) is excluded from version control via `.gitignore` — recreate it locally using the steps above.

## Future Improvements
- Add conversational memory for multi-turn interactions
- Expand intent classes for broader query handling
- Add response evaluation/feedback loop

