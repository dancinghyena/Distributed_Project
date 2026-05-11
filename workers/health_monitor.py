import threading
import time


class HealthMonitor:
    """Periodically checks worker subprocess health and restarts crashed workers (limited attempts)."""

    def __init__(self, workers, interval=1.0):
        self.workers = workers
        self.interval = interval
        self.running = False
        self.thread = None
        self.restart_counts = {}

    def start(self):
        """Starts the monitoring thread."""
        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="HealthMonitorThread"
        )
        self.thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        """Background loop to monitor worker subprocess liveness and restart on crash."""
        while self.running:
            for worker in self.workers:
                proc = worker.worker_proc
                if proc is None:
                    continue
                alive_proc = proc.is_alive()
                if alive_proc:
                    continue

                with worker.lock:
                    admin_dead = not worker.is_alive
                if admin_dead:
                    continue

                used = self.restart_counts.get(worker.id, 0)
                if used < 2:
                    self.restart_counts[worker.id] = used + 1
                    n = self.restart_counts[worker.id]
                    print(f"[HealthMonitor] Restarting worker {worker.id} (attempt {n})")
                    worker.restart()
                else:
                    print(
                        f"[HealthMonitor] Worker {worker.id} exhausted restart limit; leaving dead."
                    )
                    worker.is_alive = False

            time.sleep(self.interval)
