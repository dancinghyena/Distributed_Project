import difflib

# Small hardcoded knowledge base for the simple in-memory vector store simulation
KNOWLEDGE_BASE = [
    "Machine learning is a subset of artificial intelligence focusing on algorithms.",
    "Deep learning models require powerful GPUs for efficient training and inference.",
    "Load balancers distribute network or application traffic across multiple servers.",
    "RAG stands for Retrieval-Augmented Generation, enhancing LLMs with external data.",
    "Python is widely used in AI and building scalable distributed systems.",
    "Concurrency in Python can be managed using threads, processes, and locks.",
    "A distributed system consists of multiple software components across different nodes.",
    "Round robin, least connections, and load-aware routing are load balancing strategies."
]

def retrieve_context(query):
    """
    Simulates an in-memory vector DB retrieval using difflib for string similarity.
    Returns the top-1 matching document.
    """
    best_match = None
    highest_ratio = 0.0
    
    # Iterate over the knowledge base to find the most relevant context
    for doc in KNOWLEDGE_BASE:
        # SequenceMatcher finds the similarity ratio between query and document
        ratio = difflib.SequenceMatcher(None, query.lower(), doc.lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = doc
            
    if best_match is None or highest_ratio < 0.1:
        best_match = "No highly relevant context found in knowledge base."
        
    return best_match
