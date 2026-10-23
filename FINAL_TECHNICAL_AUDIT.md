# Final technical audit

**Audit date: 23 October 2026, Africa/Cairo.** Existing source, tests, local database, saved pipeline, vector index and reports were inspected directly. The architecture was preserved. The verdict is about a bounded, local graduate portfolio demonstration; it is not a production, cloud-deployment or Egyptian population validation certificate.

**Decision:** ready for a graduate-level local technical demonstration, with the limitations below stated explicitly. All eight requested functional competencies are implemented and verified within that scope. Broader model validity, RAG quality and production readiness remain partial. Passing mocks/offline rules alone was not accepted as evidence of live agency.

**Latest portfolio pass:** 123 automated tests and 30 offline cases passed; see [portfolio verification](reports/portfolio_verification.json) and the follow-up at the end. Earlier audit and interactive-run counts below are historical snapshots. The limited portfolio Gemini smoke is separately recorded as 1/1.

## Requirement assessment

| Requirement | Status (PASS/PARTIAL/FAIL) | File/function evidence | Test evidence | Limitation or action required |
|---|---|---|---|---|
| 1. Autonomous AI agents | **PASS** | `telecom/workflow.py: Assistant._route/_run_tools`; `telecom/gemini_llm.py: structured/choose_tool`; live mode consumes actual Gemini route labels and function arguments. | Fresh `reports/gemini_live_audit.json`: mixed request chose `both` and executed lookup, prediction and search. `tests/test_gemini_contract.py` checks SDK contracts separately. | Bounded autonomy over three read-only tools, not arbitrary execution. Initial tool use is required and permissible actions/ordering are constrained by Python. Offline routing is explicitly rule-based and proves no model autonomy. |
| 2. Multi-agent collaboration | **PASS** | `workflow.py: _customer_agent/_knowledge_agent`; `llm.py: agent_instructions`; `tools.py: ALLOWED`; customer findings are passed to knowledge planning through shared state. | Fresh mixed live case; `test_both_agents_share_and_combine_real_evidence`; Gemini contract test checks `other_agent_findings`. | Two specialisations using the same model in one process, sequentially. They are not independently trained models, distributed services or unrestricted peer conversations. |
| 3. Multi-step orchestration | **PASS** | `workflow.py: State, Assistant.__init__, chat`; conditional `StateGraph` edges; `tools.py: ToolContext/ToolExecutor.execute`. | Graph routing/clarification/failure tests, input/allowlist/ID/dependency/duplicate/budget tests; fresh mixed live sequence. | Three customer planning steps, one knowledge step, default five total tool attempts, recursion limit 12, per-call timeout. No durable checkpoints, chat memory or overall request deadline. |
| 4. REST APIs and application integration | **PASS** | `api.py: create_app/health/chat/customer/predict`; `schemas.py`; `ui.py: HTTP client and show_result`. | Full endpoint suite plus `tests/test_ui.py` actual loopback chat. `reports/final_runtime_smoke.json`: restarted servers, HTTP prediction/404/422/clarification and Streamlit render all passed. | Localhost demo with no authentication. Chat can return HTTP 200 with structured subtask errors; callers must inspect `errors`. Customer-ID binding is not user authorisation. |
| 5. Model serving and basic monitoring | **PASS — local only** | `ml.py: Predictor.predict`; `workflow.py: cached Predictor/Retriever`; `monitoring.py: event`; `api.py: request_context`; `monitor.py: summary`. | Saved model inference and real HTTP prediction passed. Log correlation/timestamp/selection/failure tests passed. Monitoring command produced real request counts and latency percentiles. | **No public cloud deployment.** CPU model/embeddings are served by local FastAPI; Gemini text generation is remote. Logs include tests and historical failures, so their percentiles are not load-test or SLA results. Single-process local logger, no alerts/dashboard. |
| 6. Hyperparameter tuning and leakage prevention | **PASS** | `train.py: train/make_pipeline`; 8 configurations × grouped 3-fold CV; preprocessing inside Pipeline; train-only F1 refit. Group hashes keep identical retained predictors together. | `test_full_grid_search_reproduces_original_benchmark_in_temporary_output` reran the whole grid and matched original scores/selection. Fold/group separation and preprocessing tests passed. | Small fixed search, one seeded outer split, no nested CV or business-cost threshold optimisation. Hash grouping only establishes isolation for identical observed feature rows. |
| 7. Reproducible ML benchmarks | **PASS** | `train.py: metrics/train`; `reports/benchmark.json`, `grid_search.csv`, `split.json`; saved `models/churn_pipeline.joblib`. Baseline and forest use the same holdout. | RF **and dummy baseline** metrics independently recomputed; seeded refit reproduced probabilities; data/bundle/provenance hashes agreed. Original model, CSV, split and reports remained byte-for-byte unchanged. | Test rows are excluded from fitting/tuning in the inspected code; this does not reconstruct an unavailable historical experiment trail. Scores apply to one public Iranian sample, not Vodafone/Egypt; uncalibrated, no causal or fairness claim. |
| 8. Generative AI, RAG and function calling | **PASS** | `rag.py: chunks/build_index/Retriever.search`; local Sentence Transformer + exact FAISS cosine search; `llm.py: grounded_answer`; `gemini_llm.py: native_history/choose_tool`. | Actual mixed Gemini answer used retrieved cancellation evidence; actual Japan-price case returned uncertainty. Source/score equality checks, semantic retrieval tests, signature preservation and invalid-citation tests passed. | 8 fictional FAQs → 11 vectors, heuristic 0.35 threshold. Citation membership and selected fact checks are **not full semantic-entailment guarantees** or broad retrieval-quality evaluation. |
| Parameterized SQLite JOIN and validation | **PASS** | `ml.py: CustomerStore.get` uses fixed `INNER JOIN ... WHERE c.customer_id = ?` and a separate tuple; read-only connections; `init_db.py` constraints and foreign key; Pydantic at entry/return. | SQL query/parameter inspection; mutations in both related tables; injection-like IDs; FK/check constraints; idempotency; explicit closure on success/failure. All 6 real stored demo rows matched fictional seeds; integrity/FK checks passed. | One-to-one profile/usage snapshot. Seeding preserves existing rows, so changing JSON does not automatically update a database record. SQLite is not a production customer-data platform. |
| Artifact/source consistency and errors | **PASS** | `Predictor.__init__` rejects corrupt/incompatible bundles; `Retriever.__init__` checks hashes/config/dimensions and reconstructed FAQ chunks; safe `AppError` handling. | Corrupt model, altered manifest excerpt, missing/stale index and invalid stored-record tests passed. `reports/final_artifact_verification.json` records current checks. | Trusted local artifacts only. Joblib validation does not make malicious pickle safe. Cached resources require a restart after rebuilding. No cryptographic signing/provenance chain is claimed. |
| Current secrets and Git ignore checks | **PASS — current files** | Gemini key remains in ignored `.env`, excluded from Settings repr; `.gitignore` covers secrets and generated artifacts; example credential blank. | Ignore checks and known-key/pattern scans of current tracked and staged content; latest evidence in `reports/portfolio_verification.json`. | Known-key/pattern scanning cannot prove absence of every possible secret. No full historical credential audit is claimed. |
| Git repository / publication | **PASS — existing repository** | Existing `main` tracks `origin/main` at the public Amrafik repository. The portfolio pass preserves its parent tree and checks the remote head before publishing. | Fetch/head comparison and staged-content verification; publication uses a normal fast-forward with an expected-parent check. | This verifies the current snapshot and update, not every prior Git object. No force-push or unrelated-history rewrite is authorised. |
| Population validity / production readiness | **PARTIAL** | Explicit caveats in Prediction, README and explanation; local-only bindings and fictional app data. | Public-data holdout and small application regression/live sets only. | No representative Egyptian/customer deployment test, calibration, fairness audit, causal study, auth, concurrency/load validation, cloud hosting or operational retention experiment. These do not block the scoped graduate demonstration. |

## Actual numerical benchmark

UCI Iranian Churn: 3,150 rows, CC BY 4.0, seed 42; **2,515 training / 635 held-out test rows**, 9 features. The 3-fold CV winner used 100 trees, unlimited depth, minimum leaf size 4 and balanced class weights; CV F1 **0.7818343626**. Fixed classification threshold **0.50**.

| Model | Precision | Recall | F1 | ROC-AUC | Confusion matrix: actual rows / predicted columns `[0,1]` |
|---|---:|---:|---:|---:|---|
| Dummy prior baseline | 0.000000 | 0.000000 | 0.000000 | 0.500000 | `[[540,0],[95,0]]` |
| Tuned Random Forest | 0.869565 | 0.842105 | 0.855615 | 0.978713 | `[[528,12],[15,80]]` |

These are real public-data results, not synthetic training results or a claim of “98% accuracy”. The synthetic app customers have no observed churn labels; their scores are illustrative inference only. The model version remains `uci563-rf-seed42-90d5fb6b`.

## Cleanup performed

- Removed the OpenAI implementation, SDK dependency, its unused `jiter` dependency, provider-selection configuration, old variables/timeouts, legacy health fields, obsolete tests and outdated failure/smoke reports. This sentence records the necessary cleanup context; active source, configuration, dependency files, scripts, tests and user guides no longer reference that provider.
- Kept the existing LangGraph nodes/edges, tool registry, RAG, FAISS, SQLite schema, ML feature set/classifier and REST/UI architecture. `llm.py` retains useful task policies/schemas; Gemini is the only SDK transport.
- Preserved the actual Gemini credential. Removed only obsolete configuration entries/comments; `.env.example` is blank of credentials. Strengthened ignore rules for credential backups. Initialised Git metadata only; no staging, commit or remote.
- Fixed explicit SQLite connection closure; removed an unused predictor cache; added safe invalid-record/model errors and index-manifest/source consistency checks.
- Retained essential no-evidence/authentication checks in Gemini tests while removing six obsolete provider-only tests. Added eleven audit checks: complete grid reproduction, baseline/hash reproduction, invalid models, SQL closure/data consistency, source substitution and Gemini safeguards. Final count: **88**, rather than the historical 83.
- Corrected documentation to Gemini-only, guarded `.env` copying against overwrite, and replaced the unavailable Windows `py` launcher command with the verified `python` executable. Added module/function explanations, agent/tool/workflow distinctions, six question walkthroughs and telecom interview practice with limitations/alternatives.

## Fresh verification results

| Check | Exact outcome | Evidence |
|---|---|---|
| Dependency compatibility | `No broken requirements found.` SDK being removed is absent from the environment. | `reports/final_command_checks.json`, audited lock file |
| Complete pytest suite | **88 passed, 0 failures/errors/skips; 32.15 seconds; 4 upstream deprecation warnings** | `reports/pytest.xml` |
| Offline evaluation | **22/22 passed**; real local components, deterministic decisions/excerpts, no Gemini calls | `reports/evaluation.json` |
| Fresh limited live Gemini | **2/2 passed** on `gemini-3.5-flash-lite` | `reports/gemini_live_audit.json` |
| Restarted servers / UI smoke | **8/8 checks passed**: health, Gemini metadata, prediction, customer 404, bad-input 422, no-tool clarification, Streamlit HTTP and render | `reports/final_runtime_smoke.json` |
| Model/index/data preservation | All 9 pre-audit artifact hashes unchanged; 100 fitted trees; model classes `[0,1]`; 11 FAISS vectors / 384 dimensions | `reports/pre_audit_artifact_hashes.json`, `final_artifact_verification.json` |
| SQL consistency | 6 profiles = 6 usage rows = 6 joined rows; integrity `ok`, no FK errors; every saved row equals its fictional seed | `reports/final_artifact_verification.json`, SQL tests |
| Secrets / command checks | Ignore rules, blank example, candidate scans and installed command/module paths passed | `reports/final_secrets_check.json`, `final_command_checks.json` |

The live mixed request executed **`get_customer → predict_churn → search_faqs`** and returned CUST-001's real local probability **0.9493094876793147** (displayed **94.9%**) with an actual cancellation source. Its policy answer described support/store requests, identity verification, outstanding balance and a possible agreement-specific fee, consistent with the retrieved fictional text. The separate Japan-price request executed search and returned uncertainty with no invented price/citation. No mocks supplied these Gemini answers.

To respect quota, the fresh audit deliberately did not repeat all seven live scenarios. The prior **7/7** run remains in `reports/gemini_live_evaluation.json` as **historical pre-cleanup evidence**. Default test mocks and the 22 offline cases are not presented as additional live autonomy checks. The actual server/UI smoke used no extra Gemini calls; the full default suite separately exercised UI chat against a real loopback FastAPI server in offline mode.

The four non-failing suite warnings come from Starlette/AnyIO, LangGraph serialization and Uvicorn/websockets. The separate Streamlit render harness also emitted a non-failing missing-ScriptRunContext warning. Dependency compatibility checks passed; no clean new-machine Windows installation, vulnerability audit or alternate Python/OS verification is claimed.

## Commands executed / reproducible checks

From `C:\Users\amror\Desktop\Telecom AI Assistant`:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q --junitxml=reports/pytest.xml
.\.venv\Scripts\python.exe -m telecom.evaluate
.\.venv\Scripts\python.exe -m telecom.evaluate_gemini --case combined --case unsupported_price
.\.venv\Scripts\python.exe -m telecom.monitor
```

Model training was reproduced by `test_full_grid_search_reproduces_original_benchmark_in_temporary_output`; it writes to a temporary folder rather than replacing the reference pipeline or exposing the test data to fitting. The documented training/build entry points resolve to existing modules. The installed interpreter, Streamlit command, API launch, endpoint requests and UI render were checked. First-time package installation/index download were not repeated unnecessarily on a clean machine.

Use separate terminals for the current local app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn telecom.api:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m streamlit run telecom/ui.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

The servers were restarted against the cleaned code and are available at localhost:8000 and localhost:8501. Do not start duplicate listeners on those ports. The consolidated evidence is `reports/final_regression_summary.json`.

## Demonstration recommendation

The project is defensible for a graduate technical demonstration if presented as **a tested local integration of bounded Gemini agents, retrieval and tabular analytics**. Show the source, trace and evidence, not just chat text. Explain the mock/offline/live distinction, the real public-data benchmark and why its high AUC is not Egyptian operational validity. Review/stage the repository yourself before publishing, then continue using only fictional customer data unless appropriate data governance and evaluation are added.

Accurate CV bullets:

- Built a Gemini-powered telecom assistant using two specialised LangGraph agents, FAISS RAG, parameterized SQLite JOINs and FastAPI/Streamlit; verified real mixed-agent tool execution and source-grounded responses.
- Tuned a Random Forest with grouped cross-validation and fold-local preprocessing on public UCI telecom data, achieving held-out F1 **0.856** and ROC-AUC **0.979** on **635** test records.
- Implemented validated REST endpoints, bounded read-only tools and structured request monitoring; passed **88 automated tests** and **22 offline evaluation cases**, with separate opt-in live verification.


## Interactive regression follow-up — 23 October 2026

Interactive use exposed a false clarification for `Show the profile for CUST-002.`. The record existed, and an unchanged live retry returned its SQL profile, indicating an intermittent model-routing error rather than a missing customer. The routing instructions now explicitly cover profile/details/usage lookups, distinguish intent from record or policy availability, and give labelled examples. Live routing and tool selection remain Gemini decisions; no deterministic live fallback was introduced.

- Complete current suite: **91 passed**, four upstream warnings, **36.89 seconds**; `reports/pytest.xml` now reflects this later run. The earlier 88-test audit timing is preserved in `reports/final_regression_summary.json`.
- Fresh live prompt checks: **10/10**, including the standard seven scenarios, a lowercase-ID usage request, selected-customer context and a repeated exact profile question. Actual registry outputs were compared with source evidence.
- Restarted localhost API: **2/2** additional live checks. The exact profile question executed `get_customer` and returned the joined SQL record. The Japan-price question executed `search_faqs` and returned uncertainty without unsupported citations. Request-correlated logs confirmed execution.
- Detailed evidence: `reports/live_prompt_regression.json`. Earlier live reports and the ML benchmark are preserved. These limited samples do not prove general routing reliability or exhaustive semantic grounding.

## Portfolio improvement follow-up — 23 October 2026

- Final complete suite: **123 passed, 0 failures/errors/skips, 36.47 seconds**, four upstream dependency warnings. Of these, 23 are marked simulated Gemini SDK contracts; they are not additional live tests.
- Expanded offline catalogue: **30/30 passed**, with real SQL/model/vector operations and zero Gemini calls. Eight retrieval cases cover all FAQ topics. The new mixed and failure scenarios verify actual call order, trace, returned source excerpts and partial results.
- The formerly failing missing-model evaluation was a setup error: the evaluator passed a factory where a predictor instance was expected. It now redirects only an isolated tool executor's predictor loader to an absent temporary file. The normal loader raises `model_unavailable`; lookup and policy retrieval survive. No production model was deleted or prediction invented.
- One completed limited Gemini scenario passed **1/1** earlier in this portfolio pass. The report observes `get_customer → predict_churn → search_faqs`, compares SQL/model outputs and checks the returned policy source. It predates only final UI wording/documentation changes; prompts, graph and model are unchanged. An interrupted two-case attempt is not counted as a completed report. Historical live reports remain distinct.
- Current localhost checks passed **7/7** without Gemini: resource readiness, SQL-equivalent profile, pipeline-equivalent prediction, missing-customer 404, invalid-input 422, Streamlit HTTP and Streamlit health. The full UI test separately sends two chats through a real loopback backend and verifies history rerender/clear.
- All nine original data/model/index/benchmark artifact hashes remain unchanged. SQLite integrity, foreign keys and six profile/usage JOIN records match the seeds. Dependency compatibility passed.
- Four owner-supplied actual interface captures were saved byte-for-byte in `docs/assets/`. They show initial, churn, roaming and combined answers with evidence panels collapsed. README captions do not present them as expanded-trace evidence.
- README and the project guide now distinguish benchmark, offline, mocked and live evidence; explain the extended cases and temporary-artifact test; and document the current sidebar labels. The removed portfolio footer remains absent.

The consolidated evidence is [reports/portfolio_verification.json](reports/portfolio_verification.json). A securely restricted free public live demo is not verified: this API lacks authentication and shared usage controls, and the two-process app needs hosting validation. No deployment was made. The [demo assessment](docs/DEMO_OPTIONS.md) recommends a short recording of local interactions.
