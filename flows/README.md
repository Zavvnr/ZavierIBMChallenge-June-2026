# flows/ — Langflow orchestration for the tactical explainer

The third commentator's Q&A can run through a **Langflow** flow (the IBM-stack
orchestration layer) instead of calling the Python engine directly. The flow is a
thin wrapper around `agent.explainer.explain()`, and the web app falls back to that
same in-process engine whenever Langflow isn't configured or reachable — so Langflow
is additive, never a hard dependency.

## Files

- `mate_explainer_component.py` — a Langflow **custom component** ("MATE Explainer")
  that calls `agent.explainer.explain()`.

## Set it up

1. Start Langflow (`pip install langflow`, then `langflow run`) **from the repo root**
   so it can import this project — or `pip install -e .` first. (Alternatively, point
   `LANGFLOW_COMPONENTS_PATH` at this `flows/` directory.)
2. In the UI, build the flow:  **ChatInput → MATE Explainer → ChatOutput**.
3. Publish the flow and copy its id.
4. Set the env vars the web app reads:

   ```dotenv
   LANGFLOW_BASE_URL=http://localhost:7860
   LANGFLOW_FLOW_ID=<your flow id>
   LANGFLOW_API_KEY=<your langflow api key>   # Langflow >= 1.5
   ```

With those set, `GET /api/ask` posts the question to the flow via
`agent.langflow_client.run_flow()` (`POST {base}/api/v1/run/{flow_id}`); unset or
failing, it answers with the in-process explainer. Either way the answer text is
produced by IBM Granite.

> Heads-up: don't name this directory `langflow/` — that would shadow the installed
> `langflow` package and break its imports.
