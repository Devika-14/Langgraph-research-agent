# 🔬 LangGraph Reflection Researcher

A **self-correcting AI research agent** built with [LangGraph](https://github.com/langchain-ai/langgraph) that autonomously searches the web, drafts a report, critiques its own output, and iteratively improves it until a quality threshold is met — with optional **Human-in-the-Loop** review at the end.

---

## ✨ Features

- 🌐 **Web Research** — Uses [Tavily Search](https://tavily.com/) to gather real-time sources
- ✍️ **Automatic Drafting** — Writes a structured report from gathered materials
- 🪞 **Self-Critique Loop** — Scores its own output (0–10) and identifies gaps
- 🔁 **Iterative Improvement** — Searches for missing info and rewrites until quality ≥ 7/10
- 🧑 **Human-in-the-Loop** — Pauses before finalising so you can accept or inject feedback
- ⚡ **Fast & Affordable** — Defaults to `llama-3.1-8b-instant` via [Groq](https://groq.com/)

---

## 🗺️ Agent Architecture

```
START
  │
  ▼
research_initial  ──►  draft  ──►  critique
                                      │
                          ┌───────────┴───────────────┐
                          │ quality met OR max iters   │ still needs work
                          ▼                            ▼
                     human_review            research_missing
                          │                            │
                          ▼                            ▼
                         END                        improve
                                                       │
                                          ┌────────────┴────────────┐
                                          │ human feedback path      │ normal path
                                          ▼                          ▼
                                     human_review               critique
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/) (free tier available)
- A [Tavily API key](https://app.tavily.com/) (free tier available)

### 2. Clone & Set Up

```bash
git clone <your-repo-url>
cd langgraph-agent

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install langgraph langchain-groq langchain-community python-dotenv tavily-python
```

### 3. Configure Environment

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

```env
# .env
GROQ_API_KEY="gsk_..."
TAVILY_API_KEY="tvly-..."

# Optional: override the default model
# GROQ_MODEL="llama-3.3-70b-versatile"
```

### 4. Run the Agent

```bash
python main.py
```

You can also pass the topic directly as a CLI argument:

```bash
python main.py "The impact of quantum computing on cryptography"
```

---

## 🧑‍💻 Usage Walkthrough

1. **Enter a research topic** — the agent begins searching the web immediately.
2. **Watch it iterate** — it will research, draft, critique, and improve automatically (up to 3 iterations or until score ≥ 7/10).
3. **Human review** — the graph pauses and displays the current draft.
   - **Option 1:** Accept the draft and finish.
   - **Option 2:** Provide specific feedback — the agent rewrites and returns for another review.

---

## ⚙️ Configuration

| Environment Variable | Default                  | Description                        |
|---------------------|--------------------------|------------------------------------|
| `GROQ_API_KEY`      | *(required)*             | Groq LLM API key                   |
| `TAVILY_API_KEY`    | *(required)*             | Tavily web search API key          |
| `GROQ_MODEL`        | `llama-3.1-8b-instant`   | Groq model to use for all LLM calls |

### Available Groq Models

| Model                        | Speed  | Quality | Token Usage |
|------------------------------|--------|---------|-------------|
| `llama-3.1-8b-instant`       | ⚡ Fast | Good    | Low         |
| `llama-3.3-70b-versatile`    | Slower | Better  | High        |

> **Tip:** If you hit Groq rate limits (429 errors), switch to `llama-3.1-8b-instant` or wait a minute before resuming.

---

## 📁 Project Structure

```
langgraph-agent/
├── main.py          # Agent graph, nodes, edges, and CLI runner
├── .env             # Your secrets (not committed)
├── .env.example     # Template for required environment variables
├── .gitignore       # Files excluded from version control
└── README.md        # This file
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

MIT
