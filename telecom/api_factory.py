"""Locally served REST API. Bind to 127.0.0.1 for this unauthenticated demo."""
import time
import uuid
from typing import Annotated

from fastapi import FastAPI, Path, Request
from fastapi.responses import JSONResponse

from telecom.config import Settings
from telecom.errors import AppError
from telecom.monitoring import event, request_id_var
from telecom.schemas import ChatRequest, ChatResponse, Customer, CustomerInput, Prediction
from telecom.workflow import Assistant


def create_app(settings: Settings | None = None, assistant: Assistant | None = None, *,
               deployment: str = "local portfolio demo") -> FastAPI:
    service = assistant or Assistant(settings)
    app = FastAPI(title="Telecom Customer Intelligence AI Assistant", version="0.1.0",
                  description="Local portfolio demo: fictional customers and DemoTel policies. No Vodafone affiliation.")
    app.state.assistant = service

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        token = request_id_var.set(request.state.request_id)
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = request.state.request_id
            response.headers["X-Response-Time-Ms"] = str(round((time.perf_counter() - started) * 1000, 3))
            return response
        finally:
            route = request.scope.get("route")
            event("http_request", method=request.method, path=getattr(route, "path", "unmatched"),
                  status_code=status, status="success" if status < 400 and not getattr(request.state, "agent_errors", False) else "error",
                  latency_ms=round((time.perf_counter() - started) * 1000, 3))
            request_id_var.reset(token)

    @app.exception_handler(AppError)
    async def application_error(request: Request, exc: AppError):
        event("application_error", error_code=exc.code, status="error")
        return JSONResponse(status_code=exc.status,
                            content={"detail": str(exc), "code": exc.code, "request_id": request.state.request_id})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        event("unexpected_error", error_type=type(exc).__name__, status="error")
        return JSONResponse(status_code=500,
                            content={"detail": "Unexpected local application error. See local logs.",
                                     "code": "internal_error", "request_id": getattr(request.state, "request_id", "")})

    @app.get("/health")
    def health():
        dependencies = {}
        for name, check in [("database", lambda: service.customers.get("CUST-001")),
                            ("churn_model", service.predictor), ("faq_index", service.retriever)]:
            try:
                check()
                dependencies[name] = "ready"
            except AppError as exc:
                dependencies[name] = exc.code
        configured = bool(service.settings.gemini_api_key)
        ready = all(value == "ready" for value in dependencies.values()) and (service.settings.app_mode == "offline" or configured)
        return JSONResponse(status_code=200 if ready else 503, content={
            "status": "ready" if ready else "degraded", "mode": service.settings.app_mode,
            "dependencies": dependencies, "llm_provider": "gemini",
            "llm_model": service.settings.gemini_model, "llm_key_configured": configured,
            "llm_connectivity": "not checked by health endpoint",
            "deployment": deployment
        })

    @app.post("/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request):
        result = service.chat(body, request.state.request_id)
        request.state.agent_errors = bool(result.errors)
        return result

    @app.get("/customers/{customer_id}", response_model=Customer)
    def customer(customer_id: Annotated[str, Path(pattern=r"^CUST-\d{3}$")]):
        return service.customers.get(customer_id)

    @app.post("/predict", response_model=Prediction)
    def predict(body: CustomerInput):
        return service.predictor().predict(service.customers.get(body.customer_id))

    return app
