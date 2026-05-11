import difflib

# Richer hardcoded knowledge base (distributed systems, GPU, fault tolerance, plus originals)
KNOWLEDGE_BASE = [
    "Machine learning is a subset of artificial intelligence focusing on algorithms.",
    "Deep learning models require powerful GPUs for efficient training and inference.",
    "Load balancers distribute network or application traffic across multiple servers.",
    "RAG stands for Retrieval-Augmented Generation, enhancing LLMs with external data.",
    "Python is widely used in AI and building scalable distributed systems.",
    "Concurrency in Python can be managed using threads, processes, and locks.",
    "A distributed system consists of multiple software components across different nodes.",
    "Round robin, least connections, and load-aware routing are load balancing strategies.",
    "GPU clusters aggregate many accelerators so large models can be served with higher throughput.",
    "CUDA and vendor libraries expose parallel kernels that map well to thousands of GPU cores.",
    "Fault tolerance uses redundancy, health checks, and automatic failover to keep services available.",
    "Distributed consensus protocols help replicas agree on state despite network delays and failures.",
    "Vector indexes such as FAISS enable fast nearest-neighbour search over dense embeddings.",
    "Sharding and replication spread data across nodes to improve capacity and resilience under load.",
]

_USE_VECTOR = False
_MODEL = None
_INDEX = None
_DOC_MATRIX = None


def _retrieve_difflib(query: str) -> str:
    best_match = None
    highest_ratio = 0.0
    for doc in KNOWLEDGE_BASE:
        ratio = difflib.SequenceMatcher(None, query.lower(), doc.lower()).ratio()
        if ratio > highest_ratio:
            highest_ratio = ratio
            best_match = doc
    if best_match is None or highest_ratio < 0.1:
        best_match = "No highly relevant context found in knowledge base."
    return best_match


try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    _DOC_MATRIX = _MODEL.encode(
        KNOWLEDGE_BASE,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    dim = _DOC_MATRIX.shape[1]
    _INDEX = faiss.IndexFlatL2(dim)
    _INDEX.add(_DOC_MATRIX.astype("float32"))
    _USE_VECTOR = True
except Exception as e:
    print(
        f"[RAG] Warning: sentence-transformers or faiss not available ({e!r}); "
        "falling back to difflib retrieval."
    )
    _MODEL = None
    _INDEX = None
    _DOC_MATRIX = None
    _USE_VECTOR = False


def retrieve_context(query: str) -> str:
    """
    Returns the top-1 matching document as a single string.
    Uses embeddings + FAISS when available; otherwise difflib over the same corpus.
    """
    if not _USE_VECTOR or _MODEL is None or _INDEX is None:
        return _retrieve_difflib(query)

    try:
        q = _MODEL.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype("float32")
        _, indices = _INDEX.search(q, 1)
        idx = int(indices[0][0])
        if 0 <= idx < len(KNOWLEDGE_BASE):
            return KNOWLEDGE_BASE[idx]
    except Exception as e:
        print(f"[RAG] Warning: Vector search failed ({e!r}); using difflib fallback.")
        return _retrieve_difflib(query)

    return _retrieve_difflib(query)
