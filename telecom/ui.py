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
    st.subheader("Customer workspace")
    selected = st.selectbox("Customer", [None] + [f"CUST-{n:03d}" for n in range(1, 7)],
                            format_func=lambda item: item or "Select a customer (optional)")
    health = backend_health(settings.backend_url)
    if health:
        service_status = "Available" if health["status"] == "ready" else "Limited availability"
        mode_label = "AI-assisted" if health["mode"] == "live" else "Offline"
        st.write(f"Service: {service_status}")
        st.write(f"Mode: {mode_label}")
        if health["mode"] == "offline":
            st.info("Offline mode uses local customer data and FAQ excerpts, without AI-generated answers.")
        else:
            st.info("Ask about plans, billing and roaming, or review a customer's churn risk.")
    else:
        st.warning("The service is currently unavailable. Please try again shortly.")
    st.markdown("**Try asking**")
    st.markdown("- How do I activate roaming?\n- Predict churn for CUST-001.\n- Explain the cancellation policy.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption("For customer questions, select a customer above or include an ID in each message.")


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

if prompt := st.chat_input("Ask about plans, billing or customer churn…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Preparing your answer…"):
            try:
                response = httpx.post(f"{settings.backend_url}/chat",
                                      json={"message": prompt, "customer_id": selected}, timeout=240)
                response.raise_for_status()
                result = response.json()
                show_result(result)
                st.session_state.messages.append({"role": "assistant", "result": result})
            except httpx.HTTPStatusError:
                st.error("We couldn't complete your request. Please try again.")
            except (httpx.HTTPError, ValueError):
                st.error("The service is unavailable or taking too long to respond. Please try again shortly.")
    st.session_state.messages = st.session_state.messages[-20:]
