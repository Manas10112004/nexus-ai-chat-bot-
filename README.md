# Nexus AI Chat Bot 🤖

A **modular AI-powered chatbot** built with **Streamlit, LangChain, Groq API, and Tavily search integration** — designed to handle conversational queries and perform data-driven analysis interactively.

**Tech Stack:**  
✔ Python • ✔ Streamlit UI • ✔ LangChain orchestration • ✔ Groq LLM • ✔ Tavily real-time search

---

## 🌟 Features

- **Interactive Chat Interface**  
  Converse with the bot through a clean Streamlit UI.

- **Multi-Model LLM Support**  
  Powered by Groq Models via LangChain for fast & context-aware responses. :contentReference[oaicite:0]{index=0}

- **Tavily Search Integration**  
  Fetch real-time web results when required for up-to-date information.

- **Modular Architecture**  
  Code is split into modules like brain, engine, database, insights & reporting.

- **Customizable & Extendable**  
  Add your own tools, connectors, or LLM providers with minimal changes.

---

## 📁 Repository Structure

```text
.
├── .devcontainer/              # Dev container configs (optional)
├── .github/workflows/          # CI workflows
├── tests/                     # Unit & integration tests
├── nexus_brain.py             # Main logic & agent orchestration
├── nexus_core.py              # Streamlit app entry point
├── nexus_engine.py            # Core processing & PSI handler
├── nexus_db.py                # Database & memory storage
├── nexus_insights.py          # Analytics & charting helpers
├── nexus_report.py            # Report export utilities
├── nexus_security.py          # Secure validation & auth flow
├── themes.py                  # UI theme definitions
├── requirements.txt           # Python dependencies
└── README.md                 # This documentation
