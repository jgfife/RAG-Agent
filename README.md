CS 6300 Vector Database Assignment

## Dataset Setup

1. Download the Wikipedia Movie Plots dataset:
   ```bash
   curl -L -o wikipedia-movie-plots.zip "https://www.kaggle.com/api/v1/datasets/download/jrobischon/wikipedia-movie-plots"
   ```
   
   Note: You may need to set up Kaggle API credentials first. See [Kaggle API documentation](https://www.kaggle.com/docs/api) for setup instructions.

2. Extract the dataset:
   ```bash
   unzip wikipedia-movie-plots.zip
   ```

## Pinecone Setup

To use Pinecone vector database functionality, you need to set your Pinecone API key:

1. Create a free account at [Pinecone](https://www.pinecone.io/)
2. Get your API key from the Pinecone console
3. Set the environment variable:
   ```bash
   export PINECONE_API_KEY="your-api-key-here"
   ```

   Or add it to your shell profile (e.g., `~/.bashrc`, `~/.zshrc`):
   ```bash
   echo 'export PINECONE_API_KEY="your-api-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Running the ChromaDB Demo

1. Install dependencies:
   ```bash
   make install
   ```

2. Run the ChromaDB program:
   ```bash
   make chroma
   ```

   Or alternatively:
   ```bash
   . .virtual_environment/bin/activate
   python3 src/chroma.py
   ```

## Running the Pinecone Demo

1. Make sure you have set your Pinecone API key (see Pinecone Setup section above)

2. Run the Pinecone program:
   ```bash
   make pinecone
   ```

   Or alternatively:
   ```bash
   . .virtual_environment/bin/activate
   python3 src/pineconeDB.py
   ```

   Both demos will:
   - Create a serverless index named "movie-plots" (if it doesn't exist)
   - Load the movie dataset into the index
   - Process queries and measure performance metrics
   - Score query relevancy using LLM evaluation
   - Calculate Information Retrieval metrics (Recall@K, NDCG@K)
   - Save detailed logs to the `logs/` directory
