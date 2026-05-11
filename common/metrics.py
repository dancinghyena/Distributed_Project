import threading
import time


class MetricsCollector:
    """Collects and aggregates performance metrics across all workers in a thread-safe manner."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.failed_requests = 0
        self.completed_so_far = 0
        self.expected_total = None
        self.latencies = []
        self.worker_stats = {}
        self.gpu_util_stats = {}
        self.start_time = None
        self.end_time = None

    def set_expected_total(self, n):
        """Total successful completions expected (used for [Progress] lines)."""
        with self.lock:
            self.expected_total = n

    def start(self):
        """Starts the timer for throughput calculation."""
        with self.lock:
            self.start_time = time.time()

    def stop(self):
        """Stops the timer for throughput calculation."""
        with self.lock:
            self.end_time = time.time()

    def record_failure(self, worker_id, reason):
        """Increments failed request counter (lock-safe)."""
        with self.lock:
            self.failed_requests += 1

    def record_gpu_util(self, worker_id, util: float):
        """Updates a running average of reported GPU utilisation per worker (0.0–1.0)."""
        with self.lock:
            if worker_id not in self.gpu_util_stats:
                self.gpu_util_stats[worker_id] = {"sum": 0.0, "n": 0}
            st = self.gpu_util_stats[worker_id]
            st["sum"] += float(util)
            st["n"] += 1

    def _maybe_print_progress_locked(self):
        exp = self.expected_total
        if not exp:
            return
        if self.completed_so_far == 0 or self.completed_so_far % 100 != 0:
            return
        elapsed = max(time.time() - (self.start_time or time.time()), 1e-6)
        tput = self.completed_so_far / elapsed
        print(
            f"[Progress] {self.completed_so_far}/{exp} requests done | "
            f"Throughput so far: {tput:.1f} req/s | Failures: {self.failed_requests}"
        )

    def record(self, worker_id, latency):
        """Records a successful request processing latency for a given worker."""
        with self.lock:
            self.total_requests += 1
            self.completed_so_far += 1
            self.latencies.append(latency)
            if worker_id not in self.worker_stats:
                self.worker_stats[worker_id] = {"count": 0, "latencies": []}
            self.worker_stats[worker_id]["count"] += 1
            self.worker_stats[worker_id]["latencies"].append(latency)
            self._maybe_print_progress_locked()

    def print_summary(self):
        """Prints a summary report of the load test performance."""
        with self.lock:
            if self.total_requests == 0 and self.failed_requests == 0:
                print("[Metrics] No requests handled.")
                return

            duration = (self.end_time or time.time()) - (self.start_time or time.time())
            throughput = self.total_requests / duration if duration > 0 else 0

            print("\n" + "=" * 40)
            print("[Metrics] --- PERFORMANCE METRICS SUMMARY ---")
            print("=" * 40)
            print(f"[Metrics] Total Requests Handled: {self.total_requests}")
            print(f"[Metrics] Failed Requests:        {self.failed_requests}")
            print(f"[Metrics] Total Time Elapsed:     {duration:.2f}s")
            print(f"[Metrics] Throughput:             {throughput:.2f} requests/second")

            if self.latencies:
                avg_latency = sum(self.latencies) / len(self.latencies)
                min_latency = min(self.latencies)
                max_latency = max(self.latencies)
                print(f"[Metrics] Overall Latency:")
                print(f"[Metrics]   - Average: {avg_latency:.4f}s")
                print(f"[Metrics]   - Minimum: {min_latency:.4f}s")
                print(f"[Metrics]   - Maximum: {max_latency:.4f}s")

            if self.gpu_util_stats:
                print("\n[Metrics] Per-Worker GPU Utilisation (avg):")
                for wid in sorted(self.gpu_util_stats.keys()):
                    st = self.gpu_util_stats[wid]
                    avg_u = st["sum"] / st["n"] if st["n"] else 0.0
                    print(f"[Metrics]   Worker {wid}: {avg_u:.3f}")

            print("\n[Metrics] Per-Worker Stats:")
            for wid, stats in sorted(self.worker_stats.items()):
                w_count = stats["count"]
                w_avg_lat = sum(stats["latencies"]) / w_count if w_count > 0 else 0
                print(f"[Metrics]   Worker {wid}: {w_count} requests, Avg Latency: {w_avg_lat:.4f}s")
            print("=" * 40 + "\n")
