# AI Marketing Research Agent

This project implements an intelligent agent for conducting marketing research and competitive analysis using Reddit data. The agent leverages NLP and LLM capabilities to analyze social media discussions, identify market trends, and generate actionable marketing insights.

## Features

- Automated Reddit data collection and analysis
- Intelligent query augmentation for comprehensive research
- Vector database storage for efficient information retrieval
- LLM-powered insight generation
- Graph-based conversation flow

## Technology Stack

- **chromadb**: Vector database for storing and managing embeddings
- **praw**: Python Reddit API Wrapper
- **openai**: OpenAI's API integration
- **python-dotenv**: Environment variable management
- **langchain**: Framework for LLM applications
- **langchain-openai**: OpenAI integration for LangChain
- **langchain-text-splitters**: Text processing utilities
- **langgraph**: Graph-based operations

## Setup

1. Install required packages:
```bash
pip install chromadb praw openai python-dotenv langchain langchain-openai langchain-text-splitters langgraph
```

2. Environment Variables
The following environment variables would be required to run the Reddit API functionality:
```
OPENAI_API_KEY=your_openai_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent
```

Note: For submission purposes, these credentials are not included and not required. Users wanting to run this code would need to obtain their own API credentials from:
- Reddit API credentials: https://www.reddit.com/prefs/apps
- OpenAI API key: https://platform.openai.com/api-keys

## Project Structure

- `AIMarketResearch.ipynb`: Main notebook containing the implementation
- `augment.json`: Query augmentation rules
- `generator_prompt.txt`: System prompt for insight generation
- `data/chroma_db/`: Directory for vector database storage

## Core Components

### Knowledge Base Generation
- Subreddit search and data collection
- Post and comment extraction
- Vector embedding generation
- ChromaDB storage

### Query Processing
- Query augmentation using predefined rules
- Vector similarity search
- Context-aware response generation

### Insight Generation
- Market trend analysis
- Competitive intelligence
- Customer pain point identification
- Strategic recommendations

## Usage

1. Set `GENERATE_KNOWLEDGE=True` to activate knowledge base generation
2. Input search query and number of subreddits to analyze
3. The agent will:
   - Search relevant subreddits
   - Collect and process posts/comments
   - Generate embeddings
   - Store information in ChromaDB
   - Generate marketing insights

## Output Format

The agent provides insights in the following structure:
1. Key Findings
2. Market Implications
3. Recommendations§
