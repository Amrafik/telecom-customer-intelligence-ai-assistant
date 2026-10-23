"""Offline private-trial entry point; local users continue to run telecom/ui.py."""
import os
import sys
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import streamlit as st

from telecom.cloud_runtime import CloudLimits, get_runtime
from telecom.conversation import previous_refusal
from telecom.monitoring import event

st.set_page_config(page_title="Telecom Customer Intelligence", page_icon="◉", layout="centered")
st.set_option("client.toolbarMode", "viewer")


@st.cache_resource
def runtime_for(limits_json):
    return get_runtime(CloudLimits.model_validate_json(limits_json))


try:
    try:
        configuration = dict(st.secrets.get("cloud", {}))
    except FileNotFoundError:
        configuration = {}
    limits = CloudLimits.model_validate(configuration)
    with st.spinner("Preparing customer data and local models…"):
        runtime = runtime_for(limits.model_dump_json()).ensure_started()
except Exception as exc:
    event("cloud_startup_failed", status="error", error_type=type(exc).__name__)
    st.error("The offline demo could not start. The owner can check the deployment guide and cloud logs.")
    st.stop()

st.title("Telecom Customer Intelligence")
st.caption("Customer insights, churn prediction and telecom support.")
st.info(
    "**Offline demo** · This hosted demo uses deterministic routing and makes no Gemini API calls. "
    "Requests execute the real SQLite customer lookup, Scikit-learn churn model, FAISS retrieval "
    "and LangGraph workflow as needed."
)
st.caption(
    "The full local version supports Gemini-powered agent routing, tool calling and grounded response "
    "generation, verified in recorded live tests. "
    "[GitHub · implementation and evaluation evidence]"
    "(https://github.com/Amrafik/telecom-customer-intelligence-ai-assistant)"
)
with st.sidebar:
    st.subheader("Customer workspace")
    selected = st.selectbox("Customer", [None] + [f"CUST-{n:03d}" for n in range(1, 7)],
                            format_func=lambda item: item or "Select a customer (optional)")
    st.write("Mode: Offline")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Include a customer ID in each customer question, or select one above.")


def show_result(result):
    st.markdown(result["answer"])
    st.caption(f"{result['route']} · Offline · {result['latency_ms'] / 1000:.2f}s")
    if result["sources"]:
        with st.expander("Sources and evidence"):
            for source in result["sources"]:
                st.markdown(f"**{source['title']}**")
                st.code(source["reference"], language=None)
                st.write(source["excerpt"])
                if source["score"] is not None:
                    st.caption(f"Retrieval similarity: {source['score']:.3f} (not a confidence probability)")
    if result["tool_trace"]:
        with st.expander("Agent workflow"):
            st.table(result["tool_trace"])
    if result["errors"]:
        st.warning("Part of this request could not be completed: " + ", ".join(result["errors"]))


if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "result" in message:
            show_result(message["result"])
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask about plans, billing or customer churn…", disabled=not limits.enabled):
    previous_intent = previous_refusal(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            response = httpx.post(runtime.url + "/chat", headers=runtime.headers,
                                  json={"message": prompt, "customer_id": selected,
                                        "previous_intent": previous_intent}, timeout=60, trust_env=False)
            if response.status_code == 429:
                st.warning("The shared demo allowance is busy or exhausted. Please try again later.")
            else:
                response.raise_for_status()
                result = response.json()
                show_result(result)
                st.session_state.messages.append({"role": "assistant", "result": result})
        except (httpx.HTTPError, ValueError):
            st.error("The service could not complete this request. Please try again shortly.")
    st.session_state.messages = st.session_state.messages[-20:]
