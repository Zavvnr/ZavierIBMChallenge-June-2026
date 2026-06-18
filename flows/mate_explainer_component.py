"""
MATE Explainer — a Langflow custom component that wraps agent.explainer.explain().

Paste this into Langflow's custom-component editor (or drop it in a folder pointed to
by LANGFLOW_COMPONENTS_PATH) to get a "MATE Explainer" node. Build the flow as:

    ChatInput  ->  MATE Explainer  ->  ChatOutput

then publish it and copy the flow id into LANGFLOW_FLOW_ID. The web app calls that
flow via agent.langflow_client; if Langflow is down, it falls back to this same
explain() in-process — so the flow is orchestration, not a hard dependency.

NOTE: Langflow must be able to import this project (run Langflow from the repo root,
or `pip install -e .`). Import paths below match Langflow 1.x; adjust if your
Langflow version differs (e.g. `from langflow.schema.message import Message`).
"""
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Message


class MateExplainerComponent(Component):
    display_name = "MATE Explainer"
    description = "Answer a football question via MATE's Granite explainer (RAG over the Laws of the Game)."
    name = "MateExplainer"
    icon = "message-circle"

    inputs = [
        MessageTextInput(name="question", display_name="Question", required=True),
        MessageTextInput(name="language", display_name="Language", value="en"),
    ]
    outputs = [Output(display_name="Answer", name="answer", method="build_answer")]

    def build_answer(self) -> Message:
        from agent.explainer import explain  # the in-process engine (retrieval + Granite)
        text = explain(self.question, language=(self.language or "en"))
        self.status = text  # shows in the Langflow node for quick debugging
        return Message(text=text)
