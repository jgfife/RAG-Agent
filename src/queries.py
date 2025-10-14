#!/usr/bin/env python3
"""Generate vector database queries from text using Ollama."""

from __future__ import annotations

import random
import requests
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class SearchResult:
    """Represents a single search result from vector database query."""
    rank: int
    distance: float
    document: str
    relevancy_score: Optional[float] = None
    relevancy_explanation: Optional[str] = None
    is_relevant: Optional[int] = None
    
    def add_relevancy(self, score: float, explanation: str) -> None:
        """Add relevancy scoring in-place."""
        self.relevancy_score = score
        self.relevancy_explanation = explanation
        self.is_relevant = 1 if score >= 4.0 else 0
    
    def __str__(self) -> str:
        """Pretty print representation of search result."""
        # Truncate document to first 80 characters for readability
        doc_preview = self.document[:80] + "..." if len(self.document) > 80 else self.document
        
        # Format relevancy info if available
        relevancy_info = ""
        if self.relevancy_score is not None:
            relevancy_info = f" | Score: {self.relevancy_score:.1f}/5.0"
            relevant_text = "Yes" if self.is_relevant == 1 else "No"
            relevancy_info += f" | Relevant: {relevant_text}"
        
        result = f"    [{self.rank}] Distance: {self.distance:.3f}{relevancy_info}\n"
        result += f"        \"{doc_preview}\""
        
        if self.relevancy_explanation:
            result += f"\n        Explanation: {self.relevancy_explanation}"
        
        return result


@dataclass 
class QueryResult:
    """Represents results for a single query."""
    query_id: int
    query: str
    latency_ms: float
    results: List[SearchResult]
    recall: Optional[float] = None
    ndcg: Optional[float] = None
    
    @classmethod
    def from_chroma_response(
        cls, 
        query_id: int, 
        query: str, 
        latency_ms: float, 
        chroma_response: Dict[str, Any]
    ) -> 'QueryResult':
        """Factory method to create QueryResult from ChromaDB response."""
        results = [
            SearchResult(
                rank=j + 1,
                distance=distance,
                document=doc
            )
            for j, (doc, distance) in enumerate(
                zip(chroma_response['documents'][0], chroma_response['distances'][0])
            )
        ]
        return cls(query_id, query, latency_ms, results)
    
    @classmethod
    def from_pinecone_response(
        cls,
        query_id: int,
        query: str,
        latency_ms: float,
        pinecone_response: Dict[str, Any]
    ) -> 'QueryResult':
        """Factory method to create QueryResult from Pinecone response."""
        results = [
            SearchResult(
                rank=j + 1,
                distance=match['score'],
                document=match['metadata']['text']
            )
            for j, match in enumerate(pinecone_response['matches'])
        ]
        return cls(query_id, query, latency_ms, results)
    
    def __str__(self) -> str:
        """Pretty print representation of query result."""
        # Header with query info
        result = f"Query #{self.query_id} (ID: {self.query_id}) [{self.latency_ms:.1f}ms]\n"
        result += f"  Query: \"{self.query}\"\n"
        
        # Add metrics if available
        metrics_info = []
        if self.recall is not None:
            metrics_info.append(f"Recall: {self.recall:.3f}")
        if self.ndcg is not None:
            metrics_info.append(f"nDCG: {self.ndcg:.3f}")
        
        if metrics_info:
            result += f"  {', '.join(metrics_info)}\n"
        
        result += f"\n  Results:\n"
        
        # Add each search result
        for search_result in self.results:
            result += str(search_result) + "\n"
        
        return result


class QueryResults:
    """Container for managing multiple query results with relevancy scoring."""
    
    def __init__(self):
        self.results: List[QueryResult] = []
    
    def add_result(self, query_result: QueryResult) -> None:
        """Add a query result to the collection."""
        self.results.append(query_result)
    
    def get_latencies(self) -> List[float]:
        """Get list of all query latencies in milliseconds."""
        return [result.latency_ms for result in self.results]
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)
    
    def score_relevancy(
        self, 
        model: str = "gpt-oss", 
        ollama_url: str = "http://localhost:11434",
        max_workers: int = 10
    ) -> None:
        """Score the relevancy of query results using Ollama with parallel processing."""
        print(f"Scoring relevancy for {len(self.results)} queries using {model} with {max_workers} parallel workers...")
        
        def score_single_result(args):
            """Score a single result - designed for parallel execution."""
            query, result, query_id, result_idx = args
            document = result.document
            
            prompt = f"""Rate the relevancy of this document to the given query on a scale of 1-5:

Query: "{query}"

Document: "{document}"

Provide your response in this exact format:
Score: [1-5]
Explanation: [brief 1-2 sentence explanation]

Where:
1 = Not relevant at all
2 = Slightly relevant and has minimal connection such as matching genre
3 = Moderately relevant and has parallels in topic or subject matter
4 = Highly relevant and has parallels in time period, location, theme, or genre
5 = Extremely relevant and directly addresses the query"""
            
            try:
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=120
                )
                response.raise_for_status()
                
                result_json = response.json()
                generated_text = result_json.get("response", "")
                
                # Parse score and explanation from response
                score = 3.0  # Default fallback score
                explanation = "Unable to parse response"
                
                for line in generated_text.strip().split('\n'):
                    line = line.strip()
                    if line.startswith("Score:"):
                        try:
                            score_str = line.replace("Score:", "").strip()
                            score = float(score_str)
                            # Clamp score to 1-5 range
                            score = max(1.0, min(5.0, score))
                        except ValueError:
                            pass
                    elif line.startswith("Explanation:"):
                        explanation = line.replace("Explanation:", "").strip()
                
                return (query_id, result_idx, score, explanation, None)
                
            except Exception as e:
                error_msg = f"Error during scoring: {str(e)}"
                return (query_id, result_idx, 3.0, error_msg, e)
        
        # Prepare all scoring tasks
        scoring_tasks = []
        for query_result in self.results:
            query = query_result.query
            for result_idx, result in enumerate(query_result.results):
                scoring_tasks.append((query, result, query_result.query_id, result_idx))
        
        print(f"Processing {len(scoring_tasks)} documents across {len(self.results)} queries...")
        
        # Execute scoring tasks in parallel
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(score_single_result, task): task for task in scoring_tasks}
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                query_id, result_idx, score, explanation, error = future.result()
                
                # Find the correct query result and add relevancy data
                for query_result in self.results:
                    if query_result.query_id == query_id:
                        query_result.results[result_idx].add_relevancy(score, explanation)
                        break
                
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(scoring_tasks):
                    print(f"Completed scoring {completed_count}/{len(scoring_tasks)} documents...")
                
                if error:
                    print(f"Error scoring query {query_id}, result {result_idx}: {error}")
        
        print(f"Completed relevancy scoring for {len(self.results)} queries using parallel processing")
    
    def calculate_recall_at_k(self) -> float:
        """
        Calculate recall@k for each query and return the averaged recall value.
        Uses k = len(query_result.results) for each query (full result set).
        
        Recall@k = (# relevant items retrieved in top-k) / (total # relevant items for query)
        
        Returns:
            Average recall value across all queries
        """
        if not self.results:
            return 0.0
            
        recall_scores = []
        
        for query_result in self.results:
            # Use full result set as k for this query
            k = len(query_result.results)
            
            # Count total relevant items for this query
            relevant_in_topk = sum(1 for result in query_result.results if result.is_relevant == 1)
            
            if relevant_in_topk == 0:
                # Store 0.0 for queries with no relevant items
                query_result.recall = 0.0
                continue
            
            recall = relevant_in_topk / k
            query_result.recall = recall
            recall_scores.append(recall)
        
        # Return average recall across all queries
        return sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    
    def calculate_ndcg_at_k(self) -> float:
        """
        Calculate nDCG@k for each query and return the averaged nDCG value.
        Uses k = len(query_result.results) for each query (full result set).
        
        nDCG@k = DCG@k / IDCG@k
        DCG@k = sum(rel_i / log2(i + 1)) for i=1 to k
        IDCG@k = DCG@k for ideally ranked results
        
        Returns:
            Average nDCG value across all queries
        """
        if not self.results:
            return 0.0
            
        ndcg_scores = []
        
        for query_result in self.results:
            # Use full result set as k for this query
            k = len(query_result.results)
            
            # Get relevance scores for all results
            relevance_scores = []
            for result in query_result.results:
                if result.is_relevant is not None:
                    relevance_scores.append(result.is_relevant)
                else:
                    relevance_scores.append(0.0)
            
            if not relevance_scores:
                query_result.ndcg = 0.0
                continue
                
            # Calculate DCG@k
            dcg = 0.0
            for i, rel_score in enumerate(relevance_scores):
                if rel_score > 0:
                    dcg += rel_score / (math.log2(i + 2))  # i+2 because positions start at 1
            
            # Calculate IDCG@k (ideal DCG with perfect ranking)
            ideal_scores = sorted(relevance_scores, reverse=True)
            idcg = 0.0
            for i, rel_score in enumerate(ideal_scores):
                if rel_score > 0:
                    idcg += rel_score / (math.log2(i + 2))
            
            # Calculate nDCG@k
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
                
            query_result.ndcg = ndcg
            ndcg_scores.append(ndcg)
        
        # Return average nDCG across all queries
        return sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    
    def __str__(self) -> str:
        """Pretty print summary of all query results."""
        if not self.results:
            return "No query results available"
        
        # Header
        result = f"Query Results Summary ({len(self.results)} queries)\n"
        result += "=" * 50 + "\n"
        
        # Performance metrics
        latencies = self.get_latencies()
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            result += f"Performance: Avg {avg_latency:.1f}ms, Min {min_latency:.1f}ms, Max {max_latency:.1f}ms\n"
        
        # IR metrics summary
        recalls = [qr.recall for qr in self.results if qr.recall is not None]
        ndcgs = [qr.ndcg for qr in self.results if qr.ndcg is not None]
        
        if recalls:
            avg_recall = sum(recalls) / len(recalls)
            result += f"Average Recall: {avg_recall:.3f}\n"
        
        if ndcgs:
            avg_ndcg = sum(ndcgs) / len(ndcgs)
            result += f"Average nDCG: {avg_ndcg:.3f}\n"
        
        result += "=" * 50
        return result
    
    def detailed_report(self) -> str:
        """Generate detailed report with all individual query results."""
        if not self.results:
            return "No query results available"
        
        result = str(self) + "\n\n"
        result += "Detailed Query Results:\n"
        result += "=" * 50 + "\n\n"
        
        for query_result in self.results:
            result += str(query_result) + "\n"
        
        return result


def print_ir_metrics(avg_recall: float, avg_ndcg: float) -> None:
    """
    Print formatted IR metrics output.
    
    Args:
        avg_recall: Average recall value across all queries
        avg_ndcg: Average nDCG value across all queries
    """
    print("Information Retrieval Metrics:")
    print(f"  Average Recall: {avg_recall:.3f}")
    print(f"  Average nDCG: {avg_ndcg:.3f}")


def generate_queries_from_dataset(
    dataset: List[Dict[str, Any]], 
    model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434"
) -> List[str]:
    """
    Generate vector database queries from randomly selected 10% of dataset using Ollama.
    
    Args:
        dataset: List of dictionaries with 'text' and 'meta' fields
        model: Ollama model to use for generation
        ollama_url: Base URL for Ollama API
        
    Returns:
        List of generated queries suitable for vector database search
    """
    
    sample_size = 500

    # Try to load existing queries from file
    queries = []
    queries_file = Path("dataset/chroma_queries.txt")
    if queries_file.exists():
        print(f"Loading queries from {queries_file}")
        with open(queries_file, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Loaded {len(queries)} queries from file:")

    # bail if we have enough
    if len(queries) >= sample_size:
        return queries
    
    print(f"Query file {queries_file} not found or wasn't complete. Generating new queries...")

    # Randomly select sample size from the dataset
    selected_items = random.sample(dataset, sample_size)
    count = len(queries)  # start from existing count if any
    with open(queries_file, 'w', encoding='utf-8') as f:
        for item in selected_items:
            text = item["text"]
            
            prompt = f"""
    Based on the following text, generate 1 search query that someone might use to find this document in a vector database. The query should be:
    - A natural language question or phrase
    - Related to the main concepts and topics in the text
    - A summary of the key ideas

    Text: {text[:500]}...  # Truncated for brevity

    Generate exactly 1 query:
    """
            
            try:
                response = requests.post(
                    f"{ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                response.raise_for_status()
                
                result = response.json()
                generated_text = result.get("response", "")
                
                # Parse query from response
                for line in generated_text.strip().split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Remove numbering if present (1., 2., etc.)
                        if line[0].isdigit() and '.' in line[:3]:
                            line = line.split('.', 1)[1].strip()
                        queries.append(line) # add to results
                        f.write(f"{line}\n") # persist to file
                        count += 1
                        print(f"Generated Query Progress: {count}/{sample_size}")                     
                        break  # Only take the first valid query
                
            except Exception as e:
                print(f"Error generating query for text: {e}")
    
    return queries