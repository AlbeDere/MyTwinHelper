# Personal Portfolio Helper

A conversational AI assistant that answers questions about its creator using RAG (Retrieval Augmented Generation) technology. The bot processes PDF documents containing personal and professional information, creating a knowledge base to provide accurate, context-aware responses about the portfolio owner.

## Features

- PDF document processing and vectorization
- Context-aware question answering
- Chat history management
- Professional response formatting
- Integration with OpenAI and Pinecone services

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone account and API key
- PDF documents containing portfolio information

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd personal-portfolio-helper
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with the following variables:
```
OPENAI_API_KEY=your_openai_api_key
INDEX_NAME=your_pinecone_index_name
PINECONE_API_KEY=your_pinecone_api_key
```

## Project Structure

```
personal-portfolio-helper/
├── bot.py              # Main chatbot implementation
├── reader.py           # PDF processing and vector database setup
├── requirements.txt    # Project dependencies
├── data/              # Directory for PDF documents
│   └── rag-training-doc.pdf
└── README.md
```

## Components

### reader.py
Handles document processing and vector database setup:
- Loads PDF documents using PyPDFLoader
- Splits text into manageable chunks
- Creates embeddings using OpenAI
- Stores vectors in Pinecone database

### bot.py
Implements the conversational interface:
- Manages chat history
- Processes user queries with context awareness
- Retrieves relevant information from vector storage
- Generates appropriate responses using OpenAI's GPT model

## Usage

1. First, process your PDF documents to create the vector database:
```bash
python reader.py
```

2. Start the chatbot:
```bash
python bot.py
```

3. Interact with the bot by typing questions. Type 'exit' to end the session.

## Technical Details

The project uses several key technologies:
- LangChain for RAG implementation
- OpenAI's embeddings and chat models
- Pinecone for vector storage
- Python's dotenv for environment management

The system follows a two-step process:
1. Document Processing: Converting PDF content into searchable vectors
2. Interactive QA: Using chat history and context-aware retrieval for accurate responses

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Albert Derevski