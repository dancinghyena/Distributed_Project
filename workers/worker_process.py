"""
GPU worker subprocess entry point: pulls tasks from a queue, runs RAG + LLM stub, puts results.
"""
import time
from concurrent.futures import ThreadPoolExecutor

# Concurrent inferences inside each worker process (throughput under heavy load).
_POOL_WORKERS = 32


def worker_main(worker_id, task_queue, result_queue, startup_event=None):
    """
    Runs in a child process. Exits cleanly on None sentinel.
    Catches exceptions and puts an error payload on the result queue.
    """
    from common.models import Request
    from llm.inference import run_llm
    from rag.retriever import retrieve_context

    def process_request(request):
        try:
            start = time.time()
            context = retrieve_context(request.query)
            result = run_llm(request.query, context)
            latency = time.time() - start
            result_queue.put(
                {
                    "id": request.id,
                    "result": result,
                    "latency": latency,
                    "worker_id": worker_id,
                }
            )
        except Exception as e:
            try:
                result_queue.put(
                    {
                        "id": request.id,
                        "error": repr(e),
                        "worker_id": worker_id,
                        "latency": 0.0,
                    }
                )
            except Exception:
                pass

    if startup_event is not None:
        startup_event.set()

    with ThreadPoolExecutor(max_workers=_POOL_WORKERS) as pool:
        while True:
            try:
                item = task_queue.get()
            except (EOFError, OSError):
                break
            if item is None:
                break
            if not isinstance(item, Request):
                try:
                    result_queue.put(
                        {
                            "id": getattr(item, "id", -1),
                            "error": "invalid_task_payload",
                            "worker_id": worker_id,
                            "latency": 0.0,
                        }
                    )
                except Exception:
                    pass
                continue
            pool.submit(process_request, item)
