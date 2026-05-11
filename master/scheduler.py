class Scheduler:
    """Entry point for requests. Uses load balancer to route to workers and collects metrics."""

    def __init__(self, load_balancer, metrics_collector):
        self.lb = load_balancer
        self.metrics = metrics_collector

    def record_failure(self, worker_id, reason):
        """Records a failed user request (e.g. timeout or exhausted retries)."""
        self.metrics.record_failure(worker_id, reason)

    def handle_request(self, request):
        response = self.lb.dispatch(request)
        self.metrics.record(response["worker_id"], response["latency"])
        util = response.get("gpu_utilization")
        if util is not None:
            self.metrics.record_gpu_util(response["worker_id"], util)
        return response
