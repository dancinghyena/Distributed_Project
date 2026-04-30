import threading

class LoadBalancer:
    """Distributes incoming requests to active workers based on the selected strategy."""
    
    def __init__(self, workers, strategy="round_robin"):
        self.workers = workers
        # Strategy can be 'round_robin', 'least_connections', or 'load_aware'
        self.strategy = strategy
        self.index = 0
        self.lock = threading.Lock()

    def get_active_workers(self):
        """Returns a list of currently alive workers."""
        active = []
        for w in self.workers:
            with w.lock:
                if w.is_alive:
                    active.append(w)
        return active

    def get_next_worker(self):
        """Selects the next worker according to the load balancing strategy."""
        with self.lock:
            active_workers = self.get_active_workers()
            if not active_workers:
                raise Exception("CRITICAL ERROR: All workers are dead. Cannot process request.")

            if self.strategy == "round_robin":
                # Find the next alive worker
                start_idx = self.index
                while True:
                    w = self.workers[self.index]
                    self.index = (self.index + 1) % len(self.workers)
                    with w.lock:
                        if w.is_alive:
                            return w
                    
                    if self.index == start_idx:
                        raise Exception("CRITICAL ERROR: All workers are dead. Cannot process request.")

            elif self.strategy == "least_connections":
                # Select the worker with the fewest active requests
                best_worker = None
                min_conn = float('inf')
                for w in active_workers:
                    with w.lock:
                        if w.active_requests < min_conn:
                            min_conn = w.active_requests
                            best_worker = w
                return best_worker

            elif self.strategy == "load_aware":
                # Select the worker with the lowest average latency
                best_worker = None
                min_lat = float('inf')
                for w in active_workers:
                    with w.lock:
                        if w.avg_latency < min_lat:
                            min_lat = w.avg_latency
                            best_worker = w
                return best_worker

            else:
                raise ValueError(f"Unknown load balancing strategy: {self.strategy}")

    def dispatch(self, request):
        """Dispatches the request to a worker. Reassigns if the chosen worker fails."""
        max_retries = 3
        for attempt in range(max_retries):
            worker = self.get_next_worker()
            try:
                return worker.process(request)
            except Exception as e:
                # Fault tolerance: Skip dead worker and retry task assignment
                # The next call to get_next_worker() will skip the newly dead worker
                pass
        
        raise Exception("Request failed after max retries due to worker failures.")
