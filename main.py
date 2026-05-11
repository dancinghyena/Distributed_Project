import random
import sys
import threading
import time

from client.load_generator import run_load_test
from common.metrics import MetricsCollector
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from workers.gpu_worker import GPUWorker
from workers.health_monitor import HealthMonitor


def failure_simulation_thread(workers, delay_range):
    """Simulates random worker failures during the load test."""
    time.sleep(random.uniform(*delay_range))
    target_worker1 = random.choice(workers)
    target_worker1.simulate_failure()

    time.sleep(random.uniform(*delay_range))
    active_workers = [w for w in workers if w.process_alive()]
    if active_workers:
        target_worker2 = random.choice(active_workers)
        target_worker2.simulate_failure()


def main():
    strategy = "least_connections"

    if len(sys.argv) > 1:
        strategy = sys.argv[1]

    print("=" * 50)
    print("DISTRIBUTED LLM INFERENCE LOAD BALANCER SIMULATION")
    print("=" * 50)
    print("Initializing system...")

    workers = [GPUWorker(i) for i in range(4)]
    for w in workers:
        w.start()

    lb = LoadBalancer(workers, strategy=strategy)
    print(f"[LB] Load Balancer strategy set to: {strategy}")

    metrics = MetricsCollector()
    metrics.set_expected_total(1000)
    scheduler = Scheduler(lb, metrics)

    health_monitor = HealthMonitor(workers, interval=0.5)
    health_monitor.start()
    print("[HealthMonitor] Health monitor started in background.")

    failure_thread = threading.Thread(
        target=failure_simulation_thread,
        args=(workers, (2.0, 4.0)),
        daemon=True,
    )
    failure_thread.start()

    print("[Client] Starting load test with 1000 concurrent users...")
    print("[Client] Please wait, simulating inference tasks...")

    metrics.start()

    try:
        run_load_test(scheduler, num_users=1000)
    except Exception as e:
        print(f"\n[Client] Load test aborted with error: {e}")

    metrics.stop()
    health_monitor.stop()

    for w in workers:
        w.shutdown()

    metrics.print_summary()


if __name__ == "__main__":
    main()
