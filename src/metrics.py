#!/usr/bin/env python3
"""
Metrics calculation utilities for query performance analysis.

Provides functions to calculate statistical percentiles from query time data.
"""

from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


def calculate_percentiles(query_times: List[float]) -> Dict[str, float]:
    """
    Calculate P50, P95, and P99 percentiles from an array of query times.
    
    Args:
        query_times: List of query execution times in seconds (or milliseconds)
        
    Returns:
        Dictionary containing P50, P95, and P99 values
        
    Raises:
        ValueError: If query_times is empty or contains non-numeric values
    """
    if not query_times:
        raise ValueError("Query times array cannot be empty")
    
    # Convert to numpy array for percentile calculation
    times_array = np.array(query_times, dtype=float)
    
    # Calculate percentiles
    p50 = np.percentile(times_array, 50)
    p95 = np.percentile(times_array, 95) 
    p99 = np.percentile(times_array, 99)
    
    return {
        'P50': p50,
        'P95': p95,
        'P99': p99
    }


def print_metrics(metrics: Dict[str, float], unit: str = "ms") -> None:
    """
    Print formatted metrics output.
    
    Args:
        metrics: Dictionary containing percentile metrics
        unit: Unit of measurement for display (default: "ms")
    """
    print(f"Query Performance Metrics:")
    print(f"  P50: {metrics['P50']:.2f} {unit}")
    print(f"  P95: {metrics['P95']:.2f} {unit}")
    print(f"  P99: {metrics['P99']:.2f} {unit}")