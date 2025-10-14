#!/usr/bin/env python3
"""
Pinecone demo: persistent collection, upsert (insert + update), query, and filters.

Run:
  python pinecone.py
"""

from __future__ import annotations
from pathlib import Path
import time
import sys
import os
from datetime import datetime
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import db
from queries import generate_queries_from_dataset, QueryResults, QueryResult, print_ir_metrics
from metrics import calculate_percentiles, print_metrics


class TeeLogger:
    """Logger that writes to both console and file, compatible with redirect_stdout."""
    
    def __init__(self, file_path: str):
        self.terminal = sys.stdout
        self.log_file = open(file_path, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if self.log_file:
            self.log_file.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def main():
    # Setup logging to both console and file
    logs_dir = Path("./logs")
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = logs_dir / f"pinecone_run_{timestamp}.log"
    
    # Create TeeLogger for dual output
    logger = TeeLogger(str(log_file_path))
    original_stdout = sys.stdout
    
    try:
        # Replace stdout with our logger
        sys.stdout = logger
        
        print("=" * 70)
        print(f"Pinecone Vector Database Demo - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file_path}")
        print("=" * 70)
        
        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")
        
        pc = Pinecone(api_key=api_key)
        
        # Initialize embedding model (same as ChromaDB uses)
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Index configuration
        index_name = "movie-plots"
        dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Create or connect to index
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )
            print(f"Created new Pinecone index: {index_name}")
        else:
            print(f"Connected to existing Pinecone index: {index_name}")
        
        index = pc.Index(index_name)
        
        # Wait for index to be ready
        time.sleep(1)
        
        movies = db.load_movie_plots_csv("dataset/wiki_movie_plots_deduped.csv")
        
        # Check if data is already loaded
        stats = index.describe_index_stats()
        total_vectors = stats['total_vector_count']
        
        if total_vectors != len(movies):
            print(f"\nDataset Loading:")
            print(f"Loading {len(movies)} items into Pinecone index '{index_name}'...")
            count = db.load_data_to_pinecone(index, movies, embedding_model)
            print(f"{count} items are contained in the Pinecone index '{index_name}'")
        else:
            print(f"\nDataset already loaded: {len(movies)} items in Pinecone index '{index_name}'")
        
        queries = generate_queries_from_dataset(movies, model="gemma3", ollama_url="http://localhost:11434")
        query_results = QueryResults()
        
        print(f"\nQuery Processing:")
        print(f"Processing {len(queries)} queries...")

        for i, query in enumerate(queries):
            start_time = time.time()
            
            # Generate embedding for query
            query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]
            
            # Query Pinecone
            res = index.query(
                vector=query_embedding.tolist(),
                top_k=5,
                include_metadata=True
            )
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
            
            # Create QueryResult object using factory method
            query_result = QueryResult.from_pinecone_response(i + 1, query, latency_ms, res)
            query_results.add_result(query_result)

        # Calculate and display query performance metrics
        query_latencies = query_results.get_latencies()
        if query_latencies:
            print(f"\nProcessed {len(query_results)} queries successfully")
            
            print(f"\nPerformance Metrics:")
            metrics = calculate_percentiles(query_latencies)
            print("=" * 50)
            print_metrics(metrics, unit="ms")
            print("=" * 50)

        # Score relevancy of all query results using LLM
        if len(query_results) > 0:
            print(f"\nRelevancy Scoring:")
            print(f"Scoring relevancy for {len(query_results)} queries...")
            query_results.score_relevancy(model="llama3.1", ollama_url="http://localhost:11434")
            
            # Calculate and display Information Retrieval metrics
            avg_recall = query_results.calculate_recall_at_k()
            avg_ndcg = query_results.calculate_ndcg_at_k()
            
            print(f"\nInformation Retrieval Metrics:")
            print("=" * 50)
            print_ir_metrics(avg_recall, avg_ndcg)
            print("=" * 50)
            
            # Print query results summary
            print(f"\nQuery Results Summary:")
            print(query_results)
            
            # Print detailed results
            print(f"\nDetailed Query Results:")
            print("=" * 70)
            for query_result in query_results:
                print(query_result)
                print("-" * 50)
        
        print(f"\nRun completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    finally:
        # Always restore original stdout and close logger
        sys.stdout = original_stdout
        logger.close()
        print(f"Output saved to: {log_file_path}")

if __name__ == "__main__":
    main()
