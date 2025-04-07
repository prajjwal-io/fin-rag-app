# Financial Research Copilot

An AI-powered investment research assistant that leverages Retrieval Augmented Generation (RAG) to analyze financial documents, SEC filings, and market data in real-time.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Financial Research Copilot is a comprehensive tool designed to assist investment professionals, analysts, and individual investors in conducting research by combining the power of vector databases, large language models, and financial domain knowledge. It can ingest and analyze various financial documents, answer natural language queries, generate research reports, and provide insights on financial metrics and sentiment.

## Features

- **Document Ingestion**
  - SEC filings (10-K, 10-Q, 8-K, etc.)
  - News articles and press releases
  - Financial statements and metrics
  - Custom document upload (PDF, DOCX, TXT)

- **Natural Language Querying**
  - Ask questions about companies in plain English
  - Query expansion with financial terminology
  - Answers grounded in source documents
  - Citation of information sources

- **Research Reports**
  - Generate comprehensive company research reports
  - Customizable report sections
  - Executive summaries
  - Source citations

- **Financial Analysis**
  - Financial metric extraction and analysis
  - Sentiment analysis of news and filings
  - Entity extraction (companies, metrics, dates)
  - Time-series analysis and comparisons

- **User Interface**
  - Web-based UI built with Streamlit
  - RESTful API for integration with other systems
  - Document upload and management
  - Research report visualization

## Architecture

The application follows a modular architecture with the following components:

1. **Data Ingestion Pipeline**: Collects and processes financial documents from various sources
2. **Document Processing Engine**: Cleans, chunks, and extracts metadata from documents
3. **Vector Database Integration**: Stores document embeddings and enables semantic search
4. **RAG Implementation**: Retrieves relevant information and generates answers
5. **Financial NLP Processing**: Extracts and analyzes financial information
6. **API Layer**: Provides access to the system's capabilities
7. **User Interface**: Enables user interaction with the system

## Technologies

- **Backend Framework**: FastAPI (Python)
- **Frontend**: Streamlit
- **Vector Database**: Pinecone
- **Embeddings**: OpenAI Text Embeddings
- **LLM**: OpenAI GPT models
- **Database**: MongoDB
- **NLP Tools**: spaCy, TextBlob, NLTK
- **Financial Data**: yfinance, Alpha Vantage, Finnhub (optional)
- **Deployment**: Docker, Docker Compose

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- API keys for OpenAI, Pinecone, and financial data sources (optional)

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-research-copilot.git
   cd financial-research-copilot
   ```

2. Create a `.env` file with your API keys (see Configuration section)

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access the application:
   - API: http://localhost:8000
   - UI: http://localhost:8501

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-research-copilot.git
   cd financial-research-copilot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. Create a `.env` file with your API keys (see Configuration section)

6. Start the API server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

7. In a separate terminal, start the Streamlit UI:
   ```bash
   streamlit run app/ui/streamlit_app.py
   ```

## Configuration

Create a `.env` file in the project root with the following variables:

```
# API Keys
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key  # optional
FINNHUB_API_KEY=your_finnhub_api_key  # optional

# Database
MONGODB_URI=mongodb://mongodb:27017/financial_research_db
MONGODB_USERNAME=admin
MONGODB_PASSWORD=password
MONGODB_DB_NAME=financial_research_db

# Application
SECRET_KEY=your_secret_key

# Model Configuration
OPENAI_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large

# SEC Edgar
SEC_USER_AGENT=Financial Research Copilot yourname@example.com
```

## Usage

### Ingesting Company Data

1. Navigate to the "Document Upload" page
2. Enter a company ticker symbol (e.g., AAPL)
3. Select the filing types to ingest
4. Click "Ingest Data"

### Uploading Custom Documents

1. Navigate to the "Document Upload" page
2. Upload a PDF, DOCX, or TXT file
3. Optionally associate it with a company ticker
4. Click "Upload"

### Querying the System

1. Navigate to the "Query Interface" page
2. Enter your question
3. Optionally specify a company ticker
4. Select content types to search
5. Click "Submit Query"

### Generating Research Reports

1. Navigate to the "Company Research" page
2. Enter a company ticker symbol
3. Select the topics to include in the report
4. Choose a time period
5. Click "Generate Research Report"

### Analyzing Sentiment

1. Navigate to the "Sentiment Analysis" page
2. Enter a company ticker symbol
3. Select the number of days to analyze
4. Click "Analyze Sentiment"

## API Documentation

The API documentation is available at `/docs` when the API server is running. The main endpoints include:

- `/api/v1/companies/{ticker}/ingest`: Ingest company data
- `/api/v1/companies/{ticker}/data`: Get company data summary
- `/api/v1/documents/ingest`: Ingest a document
- `/api/v1/documents/upload`: Upload a document
- `/api/v1/query`: Query the system
- `/api/v1/research`: Generate a research report
- `/api/v1/financial-metrics`: Analyze financial metrics
- `/api/v1/sentiment/{ticker}`: Analyze sentiment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.