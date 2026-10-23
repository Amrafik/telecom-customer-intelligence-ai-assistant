import httpx
import streamlit as st

from telecom.config import Settings

settings = Settings()
st.set_page_config(page_title="Telecom Customer Intelligence", page_icon="◉", layout="centered")
st.title("Telecom Customer Intelligence")
st.caption("Customer insights, churn prediction and telecom support.")


@st.cache_data(ttl=30)
def backend_health(url):
    try:
        response = httpx.get(f"{url}/health", timeout=40)
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


with st.sidebar:
    st.subheader("Demo settings")
    selected = st.selectbox("Fictional customer (optional)", [None] + [f"CUST-{n:03d}" for n in range(1, 7)],
                            format_func=lambda item: item or "No customer selected")
    health = backend_health(settings.backend_url)
    if health:
        st.write(f"Backend: {health['status']}")
        st.write(f"Mode: {health['mode']}")
        st.write("Provider: Gemini")
        if health["mode"] == "offline":
            st.info("Offline mode uses routing rules and retrieved excerpts. It makes no LLM provider calls.")
        else:
            st.info("Live mode uses Gemini for routing, tool selection and grounded FAQ answers.")
    else:
        st.warning("Start the FastAPI backend to chat.")
    st.caption("Examples: How do I activate roaming? · Predict churn for CUST-001 · Predict churn for CUST-001 and explain cancellation.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.caption("Each question is independent. Select or include a customer ID. No account changes are performed.")


def show_result(result):
    st.markdown(result["answer"])
    route_names = {"knowledge": "Knowledge Agent", "customer": "Customer Intelligence Agent",
                   "both": "Both agents", "clarify": "Clarification"}
    st.caption(f"{route_names[result['route']]} · {result['mode']} · {result['latency_ms'] / 1000:.2f}s")
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
            st.table([{key: value for key, value in item.items() if value is not None}
                      for item in result["tool_trace"]])
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

if prompt := st.chat_input("Ask about telecom policies or a fictional customer's churn"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking sources and customer tools…"):
            try:
                response = httpx.post(f"{settings.backend_url}/chat",
                                      json={"message": prompt, "customer_id": selected}, timeout=240)
                response.raise_for_status()
                result = response.json()
                show_result(result)
                st.session_state.messages.append({"role": "assistant", "result": result})
            except httpx.HTTPStatusError:
                st.error("The backend could not complete this request. Check its health and local logs.")
            except (httpx.HTTPError, ValueError):
                st.error("The backend is unavailable or timed out. Start FastAPI and try again.")
    st.session_state.messages = st.session_state.messages[-20:]
