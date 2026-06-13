# ⚽ MATE — Multilingual Agent Tactical Explainer

Live(-ish) multilingual football commentary plus an on-demand tactical explainer, powered by IBM Granite.

## Disclaimer
This project was built as part of the IBM SkillsBuild AI Builders June Challenge. The challenge takes inspiration from the 2026 FIFA World Cup tournament.

## 🚀 Project Title & Overview
The FIFA Men's World Cup is the most anticipated event on Earth, with an estimated billions of people watching each time the tournament is held. The experience and expectations vary for each country, yet they aim for the same goal: to win the World Cup trophy. Each country comes to the World Cup with its own strengths and weaknesses, leading to different tactical implementations on the field.

To fill in the excitement for those streaming matches from home, football commentary is a standard in many countries that broadcast the World Cup. Commentators narrate the game so people watching from home are aware of the "what" through the narrative they bring.

Despite that, commentary comes with one problem: it does not exist in every native language. And for many under-covered competitions, it does not exist at all.

MATE (Multilingual Agent Tactical Explainer) addresses this problem. It is an AI agent system that brings a more "humane" experience to viewers around the world by filling the gap caused by the absence of commentary in their language. Beyond narrating the ball, MATE adds a tactical analysis layer that identifies each event and the factors that correlate with it — for example, recognizing that an open gap in the defensive line led to a goal.

## 💡 What it does
Two-voice multilingual commentary — a lead (play-by-play) and an analyst (reaction) narrate the match natively in the user's chosen language, voiced by TTS. No translation step, so no translation errors to inherit.
A third, on-demand tactical commentator — silent until the user asks a question ("Why was the goal disallowed?", "Explain that goal"). It answers from (a) the match event and player-position data and (b) a RAG pipeline over the official Laws of the Game, so explanations are grounded in what actually happened and what the rules actually say.


## 🎯 Challenge themes addressed
- **Understanding & Explanation** — AI match explainers, tactical shift analysis.
- **Fan & Learning Experiences** — personalized multilingual World Cup companion, "teach me the game" assistant.

## 🌟 Why it matters
The honest comparison is not "AI vs. a beloved human commentator" — it is AI vs. silence, or AI vs. a feed you can't understand. This is the same rationale IBM used when deploying AI commentary at Wimbledon specifically for matches with no human commentary (see References).

## 📊 Dataset Description
Using StatsBomb Open Data — event dataTimestamped match events (passes, shots, fouls, cards, goals) as JSON, per matchThe commentary stream: the replayer emits these events in accelerated real timeStatsBomb 360 — freeze frames (available for World Cup 2022)Player positions around every eventThe tactical layer: formation shape, open gaps, defensive-line breaks — no computer vision neededIFAB Laws of the Game (PDF)The official football rulebookParsed with Docling into a RAG knowledge base so the Q&A commentator can cite the actual rule (e.g., offside)

Attribution: match data is provided free by StatsBomb under their public data user agreement, which requires crediting StatsBomb as the data source. This project uses it for non-commercial, educational/competition purposes.

## ⚙️ Environment Setup & Hardware Requirements
Prerequisites:
- Python 3.10+
- IBM Granite — either locally via Ollama (ollama pull granite3.3:8b) or hosted on watsonx.ai
- Langflow (pip install langflow) for the Q&A/RAG flow
- Docling (pip install docling) for Laws of the Game ingestion
- A Google Cloud key for Text-to-Speech (commentary voices)

## 💻 Hardware
No GPU required for the main pipeline. Granite 3.3 8B runs on a machine with ~16 GB RAM via Ollama (the 2B variant runs comfortably on 8 GB); the tactical layer is plain event/position data processing. A GPU is only relevant for the stretch-goal vision model (see Roadmap).

## 📥 Install
git clone https://github.com/Zavvnr/ZavierIBMChallenge-June-2026.git
cd ZavierIBMChallenge-June-2026
pip install -r requirements.txt
cp .env   # GRANITE endpoint/key or Ollama host, GOOGLE_TTS_API_KEY

## 🧠 System Architecture & AI Approach
Note: no model is trained from scratch in the current scope — the tactical layer reads player positions directly from StatsBomb 360 data, and Granite handles all reasoning and generation. A custom vision model is a stretch goal (see Roadmap).

## 🔀 flowchart TD
    SB[(StatsBomb events + 360 freeze frames)] --> EXT[data_extraction/]
    EXT --> REP[data_replayer/<br/>streams events in accelerated real time]
    REP --> AGENT{{Commentary agents<br/>IBM Granite}}
    AGENT -->|lead + analyst lines,<br/>native in target language| TTS[Text-to-Speech]
    TTS --> UI[Web UI]
    U([User question:<br/>"Why was the goal disallowed?"]) --> FLOW[Langflow RAG flow]
    LOTG[(Laws of the Game PDF)] --> DOC[Docling] --> KB[(Vector store)]
    KB --> FLOW
    REP -->|match state + positions| FLOW
    FLOW --> AGENT3{{Tactical explainer agent<br/>IBM Granite}} --> TTS

See flowchart.svg

## 📦 Components
- **data_extraction/** — downloads and caches StatsBomb event + 360 data for a match.
- **data_replayer/** — replays cached events in accelerated real time to simulate a live match.
- **agent/** — the Granite-powered commentary agents: pacing (not every throw-in is narrated), energy (a goal is not a back-pass), faithfulness guardrails (never invent events), and native generation in the target language; lead + analyst turn-taking.
- **context/** — tactical context built from 360 freeze frames (formation estimate, line heights, gaps) plus player/team color.
- **Langflow/** flow — routes a user question to retrieval (Laws of the Game via Docling, match state from the replayer) and then to the tactical explainer agent.
- **vision_model/** — stretch goal scaffold (empty by design for now). Due to its complexity, vision model is treated as an extra and might or might not be implemented as of now.

## 🤖 IBM technologies
- **Granite** (all agent reasoning and generation), 
- **Langflow** (Q&A/RAG orchestration), 
- **Docling** (rulebook ingestion).

## 📈 Results & Evaluation Metrics
(to be filled as the build progresses — planned metrics below)

MetricHow it's measuredResultEvent faithfulness% of generated lines consistent with the event log (no hallucinated goals/cards)TBDQ&A grounding% of tactical answers that correctly cite the relevant event/rule, on a hand-made test set of questionsTBDLatencyseconds from event emission to spoken audioTBDLanguage qualitynative-speaker spot checks per supported languageTBD

## 🔮 Inference / How to Use
### 1. Cache a match (events + 360 freeze frames)
python -m data_extraction.loader --match-id <STATSBOMB_MATCH_ID>

### 2. Build the knowledge bases (Laws of the Game via Docling, tactical context)
python -m context.build

### 3. Run the full pipeline (replayer -> agents -> TTS -> UI)
python -m web.app   # then open http://localhost:8080

Pick a language and a match in the UI. Commentary streams beside the match clock; type a question at any time and the tactical explainer answers it in the same language.

(Commands will be finalized as modules land — check --help on each module.)

## 🗺️ Roadmap
Stretch goal — vision model: a PyTorch player-detection/formation pipeline (vision_model/) so the same agents can run from a raw video feed instead of structured data. Pretrained detectors (e.g., YOLO + ByteTrack) first; custom training only if time allows. Might refer to Difusion Model Predictive Control (DMPC)
Real-time data feeds (Opta/Sportradar) instead of replayed historical matches.
More languages and dialects, voice/style personalization, accessibility variants.

## 📚 References
IBM SkillsBuild AI Builders Challenge: https://ibmskillsbuildchallenge-hub.bemyapp.com/
IBM Granite: https://www.ibm.com/granite · Langflow: https://www.langflow.org · Docling: https://docling-project.github.io/docling/
StatsBomb Open Data: https://github.com/statsbomb/open-data (+ statsbombpy)
IFAB Laws of the Game: https://www.theifab.com/laws-of-the-game-documents/
IBM watsonx AI commentary at Wimbledon: https://www.ibm.com/think/news/future-tennis-broadcasting-ai-sports-commentary · ESPN coverage

## ⚖️ License
Released under the MIT License — see LICENSE. The license file sits at the repository root with standard MIT text, so GitHub auto-detects it and shows "MIT License" in the About panel.

## 🤝 Acknowledgement
IBM SkillsBuild and BeMyApp for organizing the challenge · StatsBomb for free open data · IFAB for the publicly available Laws of the Game.
