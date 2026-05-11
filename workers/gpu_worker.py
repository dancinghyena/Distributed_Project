import threading
import time
from multiprocessing import Event, Process, Queue
from queue import Empty

from workers.worker_process import worker_main


class GPUWorker:
    """Represents a simulated GPU node; work runs in a separate process with queue IPC."""

    MAX_CONCURRENT = 4

    def __init__(self, worker_id):
        self.id = worker_id
        self.is_alive = True
        self.lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._orphans = {}

        self.active_requests = 0
        self.avg_latency = 0.0
        self.alpha = 0.2

        self.task_queue = None
        self.result_queue = None
        self.worker_proc = None

    @property
    def gpu_utilization(self) -> float:
        return min(self.active_requests / self.MAX_CONCURRENT, 1.0)

    def start(self):
        """Creates IPC queues and starts the worker subprocess."""
        with self.lock:
            if self.worker_proc is not None and self.worker_proc.is_alive():
                return
            self.task_queue = Queue()
            self.result_queue = Queue()
            startup_ready = Event()
            self.worker_proc = Process(
                target=worker_main,
                args=(self.id, self.task_queue, self.result_queue, startup_ready),
                name=f"GPUWorker-{self.id}",
                daemon=True,
            )
            self.worker_proc.start()
            if not startup_ready.wait(timeout=600):
                print(f"[Worker {self.id}] Warning: subprocess startup acknowledgement timed out.")
            self.is_alive = True

    def shutdown(self):
        """Signals the subprocess to exit and joins it."""
        with self.lock:
            proc = self.worker_proc
            tq = self.task_queue
        if tq is not None:
            try:
                tq.put(None)
            except Exception:
                pass
        if proc is not None:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=2)

    def simulate_failure(self):
        """Terminates the worker process and marks the node dead."""
        with self.lock:
            self.is_alive = False
            proc = self.worker_proc
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=3)
        print(f"\n[Worker {self.id}] CRITICAL FAILURE DETECTED!")

    def _update_ema(self, latency: float):
        with self.lock:
            if self.avg_latency == 0.0:
                self.avg_latency = latency
            else:
                self.avg_latency = (self.alpha * latency) + ((1 - self.alpha) * self.avg_latency)

    def process(self, request):
        """Sends a request to the subprocess and waits for the matching result (timeout 5s)."""
        deadline = time.time() + 5.0

        with self._pending_lock:
            if request.id in self._orphans:
                msg = self._orphans.pop(request.id)
                self._finalize_message(msg)
                if msg.get("error"):
                    raise Exception(msg.get("error"))
                util_snapshot = self.gpu_utilization
                return {
                    "id": msg["id"],
                    "result": msg["result"],
                    "latency": msg["latency"],
                    "worker_id": msg["worker_id"],
                    "gpu_utilization": util_snapshot,
                }

        with self.lock:
            if not self._process_alive_unsafe():
                raise Exception(f"Worker {self.id} is dead.")
            if self.task_queue is None or self.result_queue is None:
                raise Exception(f"Worker {self.id} is not started.")
            self.active_requests += 1

        try:
            self.task_queue.put(request)
            msg = None
            while time.time() < deadline:
                with self._pending_lock:
                    if request.id in self._orphans:
                        msg = self._orphans.pop(request.id)
                        break
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    cand = self.result_queue.get(timeout=min(0.2, max(remaining, 0.01)))
                except Empty:
                    continue
                if cand.get("id") == request.id:
                    msg = cand
                    break
                with self._pending_lock:
                    self._orphans[cand.get("id")] = cand

            if msg is None:
                raise TimeoutError(f"Worker {self.id} timed out waiting for result.")

            self._finalize_message(msg)
            if msg.get("error"):
                raise Exception(msg.get("error"))
            util_snapshot = self.gpu_utilization
            return {
                "id": msg["id"],
                "result": msg["result"],
                "latency": msg["latency"],
                "worker_id": msg["worker_id"],
                "gpu_utilization": util_snapshot,
            }
        finally:
            with self.lock:
                self.active_requests -= 1

    def _finalize_message(self, msg):
        if msg.get("error"):
            return
        lat = float(msg.get("latency", 0.0))
        self._update_ema(lat)

    def _process_alive_unsafe(self):
        return self.is_alive and self.worker_proc is not None and self.worker_proc.is_alive()

    def process_alive(self):
        """Used by load balancer / health checks: subprocess must be running and node enabled."""
        with self.lock:
            return self._process_alive_unsafe()

    def restart(self):
        """
        Starts a fresh subprocess and queues after a crash (caller enforces restart limits).
        """
        with self.lock:
            old_proc = self.worker_proc
        if old_proc is not None and old_proc.is_alive():
            old_proc.terminate()
            old_proc.join(timeout=2)
        with self.lock:
            self.task_queue = Queue()
            self.result_queue = Queue()
            startup_ready = Event()
            self.worker_proc = Process(
                target=worker_main,
                args=(self.id, self.task_queue, self.result_queue, startup_ready),
                name=f"GPUWorker-{self.id}-restarted",
                daemon=True,
            )
            self.worker_proc.start()
            if not startup_ready.wait(timeout=600):
                print(f"[Worker {self.id}] Warning: restart startup acknowledgement timed out.")
            self.is_alive = True
