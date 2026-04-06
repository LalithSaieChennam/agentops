"""Middleware to log predictions to PostgreSQL for monitoring."""

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = structlog.get_logger()


class PredictionLoggerMiddleware(BaseHTTPMiddleware):
    """Intercepts prediction responses and logs them for drift monitoring.

    Note: Primary prediction logging is done directly in the predict endpoint.
    This middleware serves as a secondary logging mechanism for audit trails.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Log prediction requests for audit
        if request.url.path.endswith("/predict") and request.method == "POST":
            logger.debug(
                "prediction_request_logged",
                path=request.url.path,
                status=response.status_code,
            )

        return response
