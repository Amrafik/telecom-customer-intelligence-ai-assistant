# Public demo options

Assessment date: 23 October 2026. No deployment, billing change or API-key change was performed.

An offline-only private Community Cloud wrapper is now provided; see [the deployment guide](STREAMLIT_CLOUD.md). It has not been deployed. This does not enable the invite-only **live** option assessed below.

## Recommendation

Use a two- to three-minute recording of the local application, supported by the real screenshots in the README and reproducible setup commands. The current code does not provide the access and global usage controls needed for a safely restricted, unattended live public demo. A recording lets a recruiter see Gemini tool selection and the returned evidence without exposing a public endpoint or consuming quota for every viewer.

This is a practical assessment of this repository, not a claim that free hosting is impossible.

| Option | Feasibility for this project | Remaining work or risk |
|---|---|---|
| Local walkthrough and recording | Recommended; the existing application already runs locally. | Record only fictional customer/policy data. Publish the finished recording separately if desired. No recording has been fabricated or published by this assessment. |
| Public offline application, no Gemini credential | Plausible, but not deployed or verified. Keeps Gemini usage at zero. | Still needs hosting for FastAPI and Streamlit, artifact build/persistence, resource checks and basic abuse controls. Clearly label rule-based routing; it does not demonstrate live LLM decisions. |
| Invite-only live demo | Possible with additional work, not ready as-is. | Authenticate viewers, protect both services, enforce a shared request/concurrency allowance, check the actual Google project tier/quotas, and provide an operator stop switch. Hosting availability remains unverified. |
| Unrestricted public live demo | Not recommended. | Anyone who can reach `/chat` could repeatedly spend the project's quota. A hidden API key does not prevent callers from invoking the server. |

## Why a hidden key is not enough

`GeminiLLM` passes the credential server-side, the Streamlit interface talks to FastAPI, and logs omit the key. These choices avoid putting the credential in browser code. For any future hosting, use the host's secret store, keep the key out of the repository and follow Google's current restriction guidance. This assessment has not inspected or changed the existing key's restrictions. [Google API-key guidance](https://ai.google.dev/gemini-api/docs/api-key).

The current tool budget is **per chat request**. Five allowed tool attempts do not limit how many different requests a visitor can submit. The API has no login, shared daily request cap or global concurrency gate. A password on only the Streamlit page would leave a separately exposed FastAPI service unprotected. These conclusions come from `telecom/api.py`, `telecom/tools.py` and `telecom/ui.py`.

Google's limits apply at project level and vary by model/tier; a fresh key is not a separate quota pool. Free-tier access and remaining allowance must be checked in the owner's AI Studio project. A configured key does not establish its billing status or guarantee free operation. No assumption about the owner's tier was used here. [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits).

## Free hosting is not a one-click match for this architecture

Streamlit Community Cloud supports selecting a Streamlit entry point and configuring secrets. This repository also needs a running FastAPI service: the UI's `BACKEND_URL=http://127.0.0.1:8000` refers to the host running Streamlit, not the developer's laptop. Deploying just `telecom/ui.py` would not automatically start that backend. This is an inference from the code's HTTP boundary and the platform's documented deployment flow. [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Community Cloud offers public/private sharing controls, but those controls would not automatically protect an independently hosted backend. The CPU embedding model, FAISS index, fitted estimator and SQLite database also need to be built and made available after a cold start; they are intentionally excluded from Git. Linux installation, free-host memory and restart behaviour have not been tested. [Streamlit sharing controls](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## Suggested recording outline

1. **0:00–0:20 — Purpose and architecture.** Explain that the assistant separates customer facts, churn inference and policy evidence. Show the local live-mode indicator without opening `.env`.
2. **0:20–0:50 — Customer lookup.** Ask `Show the profile for CUST-002.` and expand the SQLite source.
3. **0:50–1:40 — Both agents.** Ask `Predict churn for CUST-001 and explain the cancellation policy.` Show the actual score, FAQ citation and three-tool workflow.
4. **1:40–2:10 — Honest failure handling.** Ask the Japan roaming-price question and show the uncertainty response. Optionally show the missing-customer case.
5. **2:10–2:40 — Engineering evidence.** Show the held-out ML benchmark and the separate offline, mocked-SDK and real-Gemini reports. Explain the local-only and dataset-generalisation limits.

Record real interactions and retain genuine waiting time, or label any cuts. A recording is a demonstration of those runs, not a performance or reliability benchmark.
