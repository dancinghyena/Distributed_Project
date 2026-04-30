import threading
import time

class MetricsCollector:
    """Collects and aggregates performance metrics across all workers in a thread-safe manner."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.latencies = []
        self.worker_stats = {}  # worker_id -> {'count': 0, 'latencies': []}
        self.start_time = None
        self.end_time = None

    def start(self):
        """Starts the timer for throughput calculation."""
        self.start_time = time.time()

    def stop(self):
        """Stops the timer for throughput calculation."""
        self.end_time = time.time()

    def record(self, worker_id, latency):
        """Records a successful request processing latency for a given worker."""
        with self.lock:
            self.total_requests += 1
            self.latencies.append(latency)
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {'count': 0, 'latencies': []}
            self.worker_stats[worker_id]['count'] += 1
            self.worker_stats[worker_id]['latencies'].append(latency)

    def print_summary(self):
        """Prints a summary report of the load test performance."""
        with self.lock:
            if self.total_requests == 0:
                print("No requests handled.")
                return
            
            avg_latency = sum(self.latencies) / len(self.latencies)
            min_latency = min(self.latencies)
            max_latency = max(self.latencies)
            duration = (self.end_time or time.time()) - (self.start_time or time.time())
            throughput = self.total_requests / duration if duration > 0 else 0

            print("\n" + "="*40)
            print("--- PERFORMANCE METRICS SUMMARY ---")
            print("="*40)
            print(f"Total Requests Handled: {self.total_requests}")
            print(f"Total Time Elapsed:     {duration:.2f}s")
            print(f"Throughput:             {throughput:.2f} requests/second")
            print(f"Overall Latency:")
            print(f"  - Average: {avg_latency:.4f}s")
            print(f"  - Minimum: {min_latency:.4f}s")
            print(f"  - Maximum: {max_latency:.4f}s")
            
            print("\nPer-Worker Stats:")
            for wid, stats in sorted(self.worker_stats.items()):
                w_count = stats['count']
                w_avg_lat = sum(stats['latencies']) / w_count if w_count > 0 else 0
                print(f"  Worker {wid}: {w_count} requests, Avg Latency: {w_avg_lat:.4f}s")
            print("="*40 + "\n")
