import threading
import time

class HealthMonitor:
    """Periodically checks worker health in a background thread."""
    
    def __init__(self, workers, interval=1.0):
        self.workers = workers
        self.interval = interval
        self.running = False
        self.thread = None

    def start(self):
        """Starts the monitoring thread."""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True, name="HealthMonitorThread")
        self.thread.start()

    def stop(self):
        """Stops the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        """Background loop to monitor worker health."""
        while self.running:
            dead_workers = []
            for worker in self.workers:
                with worker.lock:
                    if not worker.is_alive:
                        dead_workers.append(worker.id)
            
            # Additional health logic could be added here (e.g. attempting to restart workers)
            # In this design, LB actively skips dead workers when dispatching
            time.sleep(self.interval)
