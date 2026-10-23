# Public demo options

Assessment date: 23 October 2026; updated to reflect the owner's subsequent public offline deployment. This documentation cleanup makes no deployment, billing or API-key changes.

The [public hosted demo](https://amrafik-telecom-ai.streamlit.app/) is deployed on Streamlit Community Cloud. It uses deterministic routing with real SQLite lookups, Scikit-learn churn predictions, FAISS retrieval and LangGraph execution, with **no Gemini API calls**. See [the deployment guide](STREAMLIT_CLOUD.md). This does not enable the invite-only **live** option assessed below.

Questions about capabilities or architecture and reactions to scope refusals also remain deterministic in the hosted demo. The local version uses validated Gemini intent classification for those cases, followed by short application-owned replies about implemented functions. A previous refusal-intent hint supplies limited continuity; it does not add general chat memory or affect tool permissions.

## Recommendation

Share the hosted offline demo and complement it with a two- to three-minute recording of the full local application. The local version supports Gemini-powered routing, function calling and grounded generation, with [recorded live verification](../reports/gemini_portfolio_smoke.json). Its original API does not provide the access and global usage controls needed for an unattended public Gemini service. A recording lets a recruiter see real Gemini tool selection and evidence without consuming quota for every viewer.

This is a practical assessment of this repository, not a claim that free hosting is impossible.

| Option | Feasibility for this project | Remaining work or risk |
|---|---|---|
| Local walkthrough and recording | Recommended; the existing application already runs locally. | Record only fictional customer/policy data. Publish the finished recording separately if desired. No recording has been fabricated or published by this assessment. |
| Public offline application, no Gemini credential | Deployed at the link above; keeps Gemini usage at zero. | Internal FastAPI, trusted model setup, SQLite/FAISS initialization and shared request controls are implemented. Cold starts, ephemeral storage and long-term free-host capacity remain limitations. Deterministic routing does not demonstrate live LLM decisions. |
| Invite-only live demo | Possible with additional work, not ready as-is. | Authenticate viewers, protect both services, enforce a shared request/concurrency allowance, check the actual Google project tier/quotas, and provide an operator stop switch. Hosting availability remains unverified. |
| Unrestricted public live demo | Not recommended. | Anyone who can reach `/chat` could repeatedly spend the project's quota. A hidden API key does not prevent callers from invoking the server. |

## Why a hidden key is not enough

`GeminiLLM` passes the credential server-side, the Streamlit interface talks to FastAPI, and logs omit the key. These choices avoid putting the credential in browser code. For any future hosting, use the host's secret store, keep the key out of the repository and follow Google's current restriction guidance. This assessment has not inspected or changed the existing key's restrictions. [Google API-key guidance](https://ai.google.dev/gemini-api/docs/api-key).

The original local application's tool budget is **per chat request**. Five allowed tool attempts do not limit how many different requests a visitor can submit. That local API has no viewer login, shared daily request cap or global concurrency gate. A password on only the Streamlit page would leave a separately exposed FastAPI service unprotected. The hosted offline wrapper instead uses an internal loopback API and shared concurrency/minute/day limits; it cannot enable Gemini. These distinctions are implemented in `telecom/api_factory.py`, `telecom/tools.py`, `telecom/ui.py` and `telecom/cloud_runtime.py`.

Google's limits apply at project level and vary by model/tier; a fresh key is not a separate quota pool. Free-tier access and remaining allowance must be checked in the owner's AI Studio project. A configured key does not establish its billing status or guarantee free operation. No assumption about the owner's tier was used here. [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## How the hosted wrapper handles the architecture

Streamlit Community Cloud supports selecting a Streamlit entry point and configuring secrets. The deployed entry point, `deploy/streamlit/app.py`, starts and reuses an internal FastAPI server. Deploying only the original `telecom/ui.py` would not start that backend: its `BACKEND_URL=http://127.0.0.1:8000` refers to the host running Streamlit, not the developer's laptop. [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Community Cloud offers public/private sharing controls; the current demo is public. Its backend is internal rather than independently exposed. A trusted copy of the fitted estimator and checksum manifest are committed under `deploy/streamlit/artifacts/`; generated SQLite/FAISS caches and embedding downloads are initialized separately and ignored by Git. Linux dependency checks, twenty cloud tests and ten cold-start/restart smoke checks passed before deployment. The measured 681.5 MiB Linux peak is not a managed-host memory benchmark or a long-term capacity guarantee. [Streamlit sharing controls](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## Suggested recording outline

1. **0:00–0:20 — Purpose and architecture.** Explain that the assistant separates customer facts, churn inference and policy evidence. Show the local live-mode indicator without opening `.env`.
2. **0:20–0:50 — Customer lookup.** Ask `Show the profile for CUST-002.` and expand the SQLite source.
3. **0:50–1:40 — Both agents.** Ask `Predict churn for CUST-001 and explain the cancellation policy.` Show the actual score, FAQ citation and three-tool workflow.
4. **1:40–2:10 — Honest failure handling.** Ask the Japan roaming-price question and show the uncertainty response. Optionally show the missing-customer case.
5. **2:10–2:40 — Engineering evidence.** Show the held-out ML benchmark and the separate offline, mocked-SDK and real-Gemini reports. Explain the hosted offline versus local Gemini distinction and dataset-generalisation limits.

Record real interactions and retain genuine waiting time, or label any cuts. A recording is a demonstration of those runs, not a performance or reliability benchmark.
