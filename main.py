import time
import threading
import random
import sys
from workers.gpu_worker import GPUWorker
from workers.health_monitor import HealthMonitor
from lb.load_balancer import LoadBalancer
from master.scheduler import Scheduler
from client.load_generator import run_load_test
from common.metrics import MetricsCollector

def failure_simulation_thread(workers, delay_range):
    """Simulates random worker failures during the load test."""
    # Wait a random time before killing the first worker
    time.sleep(random.uniform(*delay_range))
    target_worker1 = random.choice(workers)
    target_worker1.simulate_failure()
    
    # Wait a bit and kill another one just to test resilience
    time.sleep(random.uniform(*delay_range))
    active_workers = [w for w in workers if w.is_alive]
    if active_workers:
        target_worker2 = random.choice(active_workers)
        target_worker2.simulate_failure()

def main():
    # Strategy selection (you can change this to test different strategies)
    # Available options: "round_robin", "least_connections", "load_aware"
    strategy = "least_connections"
    
    if len(sys.argv) > 1:
        strategy = sys.argv[1]

    print("="*50)
    print("DISTRIBUTED LLM INFERENCE LOAD BALANCER SIMULATION")
    print("="*50)
    print("Initializing system...")
    
    workers = [GPUWorker(i) for i in range(4)]  # simulate 4 GPUs
    
    lb = LoadBalancer(workers, strategy=strategy)
    print(f"Load Balancer strategy set to: {strategy}")
    
    metrics = MetricsCollector()
    scheduler = Scheduler(lb, metrics)
    
    # Start health monitor
    health_monitor = HealthMonitor(workers, interval=0.5)
    health_monitor.start()
    print("Health monitor started in background.")

    # Start a thread to kill workers halfway through the test
    failure_thread = threading.Thread(
        target=failure_simulation_thread, 
        args=(workers, (2.0, 4.0)), 
        daemon=True
    )
    failure_thread.start()

    print("Starting load test with 1000 concurrent users...")
    print("Please wait, simulating inference tasks...")
    
    metrics.start()
    
    # Run test
    try:
        run_load_test(scheduler, num_users=1000)
    except Exception as e:
        print(f"\nLoad test aborted with error: {e}")
        
    metrics.stop()

    health_monitor.stop()

    # Print final summary
    metrics.print_summary()

if __name__ == "__main__":
    main()
