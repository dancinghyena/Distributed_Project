import threading
import random
from common.models import Request

def simulate_user(scheduler, user_id):
    """Simulates a single user sending a request."""
    query_topics = ["load balancing", "machine learning", "python concurrency", "deep learning gpu", "RAG"]
    topic = random.choice(query_topics)
    
    request = Request(id=user_id, query=f"Tell me about {topic}")
    
    try:
        scheduler.handle_request(request)
        # Suppress individual prints to avoid cluttering stdout during large test
        # print(f"[Client] Response {response['id']} | Latency: {response['latency']:.3f}s")
    except Exception as e:
        pass # Ignore failure prints in heavy load tests unless debugging

def run_load_test(scheduler, num_users=1000):
    """Spawns concurrent threads to simulate a heavy workload."""
    threads = []
    
    # Spawn threads
    for i in range(num_users):
        t = threading.Thread(target=simulate_user, args=(scheduler, i), name=f"UserThread-{i}")
        threads.append(t)
        t.start()
        
    # Wait for all simulated users to finish
    for t in threads:
        t.join()
