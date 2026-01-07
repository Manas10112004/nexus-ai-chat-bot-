# Nexus AI ⚡

**A Modular, Agentic Data Analysis Platform**

Nexus AI is an advanced AI agent system designed to perform autonomous data analysis, visualization, and reporting. Built on a modern Python stack (`LangChain`, `LangGraph`, `Streamlit`), it bridges the gap between raw data and actionable insights by allowing users to interact with their data using natural language.

---

## 🏗️ Architecture

Nexus AI follows a modular, agent-centric architecture designed for scalability and extensibility:

* **Frontend**: Streamlit (Python-based UI)
* **Orchestration**: LangGraph (Stateful agent workflows)
* **LLM Engine**: Groq (Llama 3.3 70B & 8B)
* **Tools**:
    * **Python Engine**: Sandboxed execution environment for Pandas/Matplotlib.
    * **Tavily Search**: Real-time web information retrieval.
* **Storage**: Supabase (Cloud database for session history & settings).

---

## 🚀 Key Features

* **Autonmous Analysis**: Upload a CSV/Excel file and ask questions like *"Find the outliers in revenue and plot the trend."*
* **Smart Visualization**: Automatically generates, renders, and saves matplotlib/seaborn charts.
* **Professional Reporting**: Exports full session history and generated charts into a downloadable PDF report.
* **Multi-Model Intelligence**: Dynamically switches between "Smart" (Llama-70b) and "Fast" (Llama-8b) models to optimize for logic vs. speed.
* **Secure & Persistent**: Includes user authentication and cloud-based session history.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.9+
* Groq API Key
* Tavily API Key
* Supabase URL & Key

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/nexus-ai.git](https://github.com/your-username/nexus-ai.git)
cd nexus-ai
```
### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Configure Secrets
Create a .streamlit/secrets.toml file (or set environment variables):
```Ini, TOML
[secrets]
GROQ_API_KEYS = "gsk_..."
TAVILY_API_KEYS = "tvly-..."
SUPABASE_URL = "[https://your-project.supabase.co](https://your-project.supabase.co)"
SUPABASE_KEY = "your-anon-key"
```
### 4.Run the Application
```bash
streamlit run nexus_core.py
```
### Built it by [Your Name]
```
**Action Item:** Once you have updated `nexus_core.py` and created `README.md`, your project will have officially moved from "Prototype" to "Documented Pre-Production" status!
```