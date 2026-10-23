import json
from dataclasses import dataclass

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from telecom.config import Settings
from telecom.errors import AppError
from telecom.llm import EvidenceLLM, agent_instructions
from telecom.monitoring import event


@dataclass(frozen=True)
class GeminiToolCall:
    name: str
    arguments: dict
    call_id: str


def native_history(history: list) -> list[types.Content]:
    """Keep complete native model parts (including thought signatures), then attach tool results."""
    result = []
    pending = None
    for item in history:
        if isinstance(item, types.Content):
            result.append(item)
            calls = [part.function_call for part in item.parts or [] if part.function_call]
            if calls:
                pending = calls[0]
        elif isinstance(item, dict) and item.get("type") == "function_call_output" and pending:
            result.append(types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                name=pending.name, id=pending.id, response=json.loads(item["output"])
            ))]))
            pending = None
        else:
            raise AppError("Invalid Gemini conversation history.", "llm_invalid_output")
    return result


class GeminiLLM(EvidenceLLM):
    def __init__(self, settings: Settings, *, client=None):
        if not settings.gemini_api_key:
            raise AppError("Live Gemini mode needs GEMINI_API_KEY. Offline mode remains available.", "api_key_missing")
        self.settings = settings
        # Explicit key prevents GOOGLE_API_KEY from silently overriding GEMINI_API_KEY.
        self.client = client or genai.Client(api_key=settings.gemini_api_key, vertexai=False,
            http_options=types.HttpOptions(timeout=int(settings.gemini_timeout_seconds * 1000),
                                           retry_options=types.HttpRetryOptions(attempts=1)))

    def _generate(self, contents, config):
        try:
            return self.client.models.generate_content(model=self.settings.gemini_model, contents=contents, config=config)
        except errors.APIError as exc:
            code = {400: "llm_request_rejected", 401: "llm_authentication_failed", 403: "llm_access_denied",
                    404: "llm_model_unavailable", 429: "llm_rate_limited"}.get(exc.code, "llm_unavailable")
            event("llm_provider_error", provider="gemini", error_code=code, http_status=exc.code)
            raise AppError("Gemini request failed. Check credentials, model access, quota and network.", code) from None
        except httpx.HTTPError as exc:
            event("llm_provider_error", provider="gemini", error_code="llm_unavailable", error_type=type(exc).__name__)
            raise AppError("Gemini network request failed or timed out.", "llm_unavailable") from None
        except ValueError:
            raise AppError("Gemini returned invalid structured output.", "llm_invalid_output") from None

    def _config(self, **kwargs):
        # Gemini 3 Flash needs a valid thinking level; keep responses small and predictable.
        thinking = types.ThinkingConfig(thinking_level="LOW") if self.settings.gemini_model.startswith("gemini-3") else None
        return types.GenerateContentConfig(max_output_tokens=2048, temperature=0, thinking_config=thinking,
                                           automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True), **kwargs)

    def structured(self, instructions: str, payload: dict, schema: type[BaseModel]):
        response = self._generate(json.dumps(payload), self._config(system_instruction=instructions,
            response_mime_type="application/json", response_json_schema=schema.model_json_schema()))
        try:
            return schema.model_validate_json(response.text or "")
        except (ValueError, TypeError):
            raise AppError("Gemini returned invalid structured output.", "llm_invalid_output") from None

    def choose_tool(self, agent: str, question: str, customer_id: str | None, history: list,
                    specifications: list[dict], prior_findings: str = ""):
        declarations = [types.FunctionDeclaration(name=spec["name"], description=spec["description"],
                                                 parameters_json_schema=spec["parameters"]) for spec in specifications]
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=json.dumps({
            "question": question, "customer_id": customer_id, "other_agent_findings": prior_findings
        }))])] + native_history(history)
        config = self._config(system_instruction=agent_instructions(agent),
            tools=[types.Tool(function_declarations=declarations)],
            tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO" if history else "ANY")))
        response = self._generate(contents, config)
        if not response.candidates or not response.candidates[0].content:
            raise AppError("Gemini did not return a usable agent response.", "llm_invalid_output")
        content = response.candidates[0].content
        calls = [part.function_call for part in content.parts or [] if part.function_call]
        if len(calls) > 1 or (not history and not calls):
            raise AppError("Gemini must return exactly one tool call for the initial agent step.", "llm_invalid_output")
        if not calls:
            return None, [content]
        call = calls[0]
        normalized = GeminiToolCall(name=call.name, arguments=call.args or {}, call_id=call.id or f"gemini-step-{len(history)}")
        return normalized, [content]
