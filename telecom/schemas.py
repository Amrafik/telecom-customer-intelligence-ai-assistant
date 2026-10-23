from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

CustomerID = Annotated[str, StringConstraints(pattern=r"^CUST-\d{3}$")]
Intent = Literal["knowledge", "customer", "both", "clarify", "out_of_scope",
                 "capabilities", "architecture", "refusal_followup"]
RefusalIntent = Literal["out_of_scope", "refusal_followup"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Customer(StrictModel):
    customer_id: CustomerID
    display_name: str
    fictional: Literal[True]
    call_failures: int = Field(ge=0, le=1000)
    complaints: Literal["yes", "no"]
    subscription_months: int = Field(ge=0, le=600)
    charge_band: int = Field(ge=0, le=9)
    seconds_of_use: int = Field(ge=0, le=10_000_000)
    calls: int = Field(ge=0, le=100_000)
    sms: int = Field(ge=0, le=100_000)
    distinct_called_numbers: int = Field(ge=0, le=100_000)
    tariff_plan: Literal["pay_as_you_go", "contract"]


class CustomerInput(StrictModel):
    customer_id: CustomerID


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=2000)
    customer_id: CustomerID | None = None
    previous_intent: RefusalIntent | None = None


class Prediction(StrictModel):
    customer_id: CustomerID
    churn_probability: float = Field(ge=0, le=1)
    predicted_churn: bool
    threshold: float = 0.5
    model_version: str
    caveat: str = "Illustrative score for a fictional customer using a public Iranian dataset; not validated for Egypt or operational decisions."


class Source(StrictModel):
    source_id: str
    title: str
    reference: str
    excerpt: str
    score: float | None = None


class ToolTrace(StrictModel):
    agent: str
    tool: str
    status: Literal["success", "error"]
    code: str | None = None


class ChatResponse(StrictModel):
    answer: str
    route: Literal["knowledge", "customer", "both", "clarify"]
    intent: Intent
    mode: Literal["offline", "live"]
    sources: list[Source] = Field(default_factory=list)
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    request_id: str
    latency_ms: float
