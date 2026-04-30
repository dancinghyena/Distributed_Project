import time
import random

def run_llm(query, context):
    """Simulates LLM inference latency."""
    # Simulate GPU inference delay with slight variance across requests
    delay = 0.1 + random.uniform(0.0, 0.2)
    time.sleep(delay)
    return f"LLM Answer to '{query}' using [{context}]"
