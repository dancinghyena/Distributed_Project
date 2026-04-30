class Scheduler:
    """Entry point for requests. Uses load balancer to route to workers and collects metrics."""
    
    def __init__(self, load_balancer, metrics_collector):
        self.lb = load_balancer
        self.metrics = metrics_collector

    def handle_request(self, request):
        # Route the request using the load balancer
        response = self.lb.dispatch(request)
        
        # Record metrics upon successful response
        self.metrics.record(response["worker_id"], response["latency"])
        return response
