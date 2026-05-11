import random
import threading

from common.models import Request


def simulate_user(scheduler, user_id):
    """Simulates a single user sending a request."""
    query_topics = [
        "load balancing",
        "machine learning",
        "python concurrency",
        "deep learning gpu",
        "RAG",
    ]
    topic = random.choice(query_topics)

    request = Request(id=user_id, query=f"Tell me about {topic}")

    try:
        scheduler.handle_request(request)
    except Exception as e:
        scheduler.record_failure(None, str(e))


def run_load_test(scheduler, num_users=1000):
    """Spawns concurrent threads to simulate a heavy workload."""
    threads = []
    for i in range(num_users):
        t = threading.Thread(
            target=simulate_user, args=(scheduler, i), name=f"UserThread-{i}"
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
