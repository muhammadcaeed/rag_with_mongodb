# RAG with MongoDB Atlas & LlamaIndex

This repository provides a clean implementation of a **Retrieval-Augmented Generation (RAG)** system using **MongoDB Atlas** as a Vector Database and **LlamaIndex** for orchestration.

## Features
- **Vector Ingestion**: Process local documents and store them as high-dimensional vectors in MongoDB.
- **Semantic Retrieval**: Uses MongoDB Atlas Vector Search to find relevant context.
- **Contextual Generation**: Powered by OpenAI to provide accurate answers based on your private data.

## Project Structure
- `load_data.py`: Script to initialize the vector store and upload your document embeddings.
- `rag.py`: The query engine used to ask questions against the indexed data.
- `key_param.py`: Configuration file for API keys and connection strings.

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- A MongoDB Atlas Cluster (M0 tier or higher).
- An OpenAI API Key.

### 2. Configure MongoDB Vector Search
You must create a Vector Search index in your MongoDB Atlas dashboard. Use the name **vector_index** and ensure the path is set to **embedding** with **1536** dimensions (for OpenAI embeddings) using **cosine** similarity.

### 3. Installation
Run the following commands to set up your environment:
- git clone https://github.com/muhammadcaeed/rag_with_mongodb.git
- cd rag_with_mongodb
- pip install llama-index pymongo llama-index-vector-stores-mongodb llama-index-embeddings-openai

### 4. Configuration
Update your `key_param.py` file with your specific credentials including your MONGO_URI, OPENAI_API_KEY, and the specific Database and Collection names you intend to use.

## Usage

1. **Load your data**: Place your source documents in a folder named `/data` and run `python load_data.py`.
2. **Query the system**: Once indexed, run `python rag.py` to start the interactive query engine.
