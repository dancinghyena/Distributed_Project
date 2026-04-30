import time
import threading
from llm.inference import run_llm
from rag.retriever import retrieve_context

class GPUWorker:
    """Represents a simulated GPU node processing LLM requests with RAG."""
    
    def __init__(self, id):
        self.id = id
        # Fault tolerance: flag to indicate if worker is available
        self.is_alive = True
        
        # Concurrency safety: lock for shared state mutations
        self.lock = threading.Lock()
        
        # Load-balancing tracking (Least Connections)
        self.active_requests = 0
        
        # Load-aware tracking (Exponential Moving Average latency)
        self.avg_latency = 0.0
        self.alpha = 0.2  # Smoothing factor

    def simulate_failure(self):
        """Simulates node failure."""
        with self.lock:
            self.is_alive = False
            print(f"\n[!] [Worker {self.id}] CRITICAL FAILURE DETECTED!")

    def process(self, request):
        """Processes the request, simulating RAG retrieval and LLM inference."""
        with self.lock:
            if not self.is_alive:
                raise Exception(f"Worker {self.id} is dead.")
            self.active_requests += 1

        try:
            start = time.time()
            
            # Simulate processing
            context = retrieve_context(request.query)
            result = run_llm(request.query, context)
            
            latency = time.time() - start
            
            with self.lock:
                # Update exponential moving average of latency
                if self.avg_latency == 0.0:
                    self.avg_latency = latency
                else:
                    self.avg_latency = (self.alpha * latency) + ((1 - self.alpha) * self.avg_latency)
            
            return {"id": request.id, "result": result, "latency": latency, "worker_id": self.id}
        finally:
            with self.lock:
                self.active_requests -= 1
