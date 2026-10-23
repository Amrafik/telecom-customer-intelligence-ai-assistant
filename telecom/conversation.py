"""Bounded scope replies and deterministic classification for offline mode."""
import re

from telecom.schemas import RefusalIntent

KNOWLEDGE_WORDS = r"\b(plan|plans|bill|billing|roaming|roam|cancel|cancellation|payment|support|faq|allowance|policy|policies|minutes|charge|charges|complaint|refund|discount|data)\b"
CUSTOMER_WORDS = r"\b(customer|profile|churn|risk|predict|prediction|leaving|likelihood)\b|CUST-"
REACTION = r"\b(why|why not|how come|refus\w*|won.t|wouldn.t|can.t|cannot|didn.t|but|seriously)\b"


def offline_route(message: str, previous_intent: RefusalIntent | None = None) -> str:
    """Rules are deliberately limited; they do not emulate Gemini understanding."""
    if re.search(r"\b(how (do|does) (you|this|the assistant|it) work|architecture|under the hood|tech(?:nology)? stack)\b", message, re.I):
        return "architecture"
    if re.search(r"\b(what can you (do|help)|your capabilities|what (is|are) (this assistant|your (purpose|tools))|who are you|can you (browse|access the internet))\b", message, re.I):
        return "capabilities"
    customer = bool(re.search(CUSTOMER_WORDS, message, re.I))
    knowledge = bool(re.search(KNOWLEDGE_WORDS, message, re.I))
    if customer or knowledge:
        return "both" if customer and knowledge else "customer" if customer else "knowledge"
    if re.search(REACTION, message, re.I) and (previous_intent or re.search(r"\b(refus\w*|answer|help|respond)\b", message, re.I)):
        return "refusal_followup"
    if re.fullmatch(r"\s*(hi|hello|hey|help|thanks|thank you|ok|okay|why|why not)[.!?\s]*", message, re.I):
        return "clarify"
    return "out_of_scope"


def scope_reply(intent: str, mode: str) -> str:
    replies = {
        "out_of_scope": "That falls outside this telecom assistant's scope. I can help with telecom policies, demo customer profiles and churn estimates.",
        "capabilities": "I can explain the demo telecom FAQs, look up a fictional customer's profile and estimate churn risk. I can't browse the web, access real accounts or make account changes.",
        "architecture": "Streamlit sends your question to FastAPI. LangGraph coordinates SQLite customer lookup, a saved Scikit-learn Random Forest for churn, and Sentence Transformers with FAISS for FAQ retrieval.",
        "refusal_followup": "I keep this assistant focused on telecom support and customer intelligence. Try a telecom question, or ask what I can do or how I work.",
        "clarify": "What would you like to explore: telecom policies, a demo customer's profile, churn risk, or how this assistant works?",
    }
    answer = replies.get(intent, replies["clarify"])
    if intent in {"capabilities", "architecture"}:
        answer += (" In this local live session, Gemini classifies intent, selects validated tools and generates grounded FAQ answers."
                   if mode == "live" else " This session uses deterministic rules and retrieved excerpts; it makes no Gemini API calls.")
    return answer if mode == "live" else "Offline mode (deterministic rules): " + answer


def previous_refusal(messages: list[dict]) -> RefusalIntent | None:
    """Use only the latest completed assistant turn; never carry records or text."""
    if not messages or messages[-1].get("role") != "assistant":
        return None
    intent = messages[-1].get("result", {}).get("intent")
    return intent if intent in {"out_of_scope", "refusal_followup"} else None
