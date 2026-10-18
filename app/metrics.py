"""Prometheus metrics, served on a separate port (never on the public API port).

Labels are kept to a small fixed set on purpose: the route template (never the raw path),
the method, the status code and a few enums. Never put a question, a document id, a tag,
a key or an error message in a label: every distinct value is another stored time series.
"""

import logging
import time
from wsgiref.simple_server import WSGIServer

from prometheus_client import CollectorRegistry, Counter, Histogram, start_http_server

logger = logging.getLogger("rag_api")

# Our own registry, not the global one, so nothing else in the process leaks into the
# output and tests can never hit duplicate-registration errors.
REGISTRY = CollectorRegistry()

UNMATCHED_ROUTE = "unmatched"

http_requests = Counter(
    "rag_http_requests_total",
    "HTTP requests handled, by method, route template and status code",
    ["method", "route", "status"],
    registry=REGISTRY,
)
http_duration = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request duration in seconds, by method and route template",
    ["method", "route"],
    registry=REGISTRY,
)
# Mutually exclusive final results of POST /query that got past validation:
#   cache_hit        answered from the answer cache
#   cache_miss       the cache was consulted, missed, and the question was answered
#   cache_disabled   the cache is off (ANSWER_CACHE_TTL_SECONDS=0) and the question was answered
#   budget_rejected  the daily budget was used up (429)
# A query that fails with a 503 after passing the budget is in none of these; see the
# 503 count in rag_http_requests_total and rag_upstream_errors_total.
query_outcomes = Counter(
    "rag_query_outcomes_total",
    "Final results of /query requests that passed validation",
    ["outcome"],
    registry=REGISTRY,
)
cache_errors = Counter(
    "rag_cache_errors_total",
    "Answer cache errors that were swallowed (fail-open)",
    ["operation"],
    registry=REGISTRY,
)
upstream_errors = Counter(
    "rag_upstream_errors_total",
    "Upstream failures turned into a generic 503",
    ["service"],
    registry=REGISTRY,
)

# Pre-create the fixed label values so every series reads 0 from the first scrape;
# rate() behaves better than with series that only appear on first use.
for _outcome in ("cache_hit", "cache_miss", "cache_disabled", "budget_rejected"):
    query_outcomes.labels(_outcome)
for _operation in ("get", "put"):
    cache_errors.labels(_operation)
for _service in ("voyage", "anthropic", "database", "other"):
    upstream_errors.labels(_service)


class MetricsMiddleware:
    """Pure ASGI middleware: counts and times every HTTP request.

    The route label is the matched route's path template (for example
    `/documents/{document_id}`), so ids never become label values. Requests that match no
    route share one label, so scanners probing random paths cannot grow the series count.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        status = 500  # what is recorded if the app raises before sending a response
        started = time.perf_counter()

        async def send_wrapper(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # The router fills scope["route"] while handling the request, so read it last.
            route = getattr(scope.get("route"), "path", None) or UNMATCHED_ROUTE
            method = scope["method"]
            http_requests.labels(method, route, str(status)).inc()
            http_duration.labels(method, route).observe(time.perf_counter() - started)


def start_metrics_server(port: int) -> WSGIServer | None:
    """Serve REGISTRY on `port` from a daemon thread. Returns the server, or None.

    Metrics are optional: if the port cannot be bound, log it and carry on so the API
    still starts.
    """
    try:
        server, _thread = start_http_server(port, registry=REGISTRY)
    except OSError as exc:
        logger.error("Metrics server could not start on port %s: %s", port, exc)
        return None
    logger.info("Metrics listening on :%s/metrics", server.server_port)
    return server


def stop_metrics_server(server: WSGIServer | None) -> None:
    if server is not None:
        server.shutdown()
        server.server_close()
