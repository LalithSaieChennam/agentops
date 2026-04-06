# AgentOps — Agentic MLOps Pipeline

[![CI](https://github.com/yourusername/agentops/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/agentops/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**Autonomous ML model monitoring, retraining, and deployment powered by LangGraph agents.**

AgentOps orchestrates 4 autonomous agents that monitor a deployed DistilBERT classifier, detect data drift and performance degradation, automatically retrain when quality drops, and deploy updated models with zero downtime.

## Architecture

```
                    ┌──────────────┐
                    │   Incoming   │
                    │   Tickets    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   FastAPI    │──── /predict ──── Prometheus
                    │   + MCP      │──── /metrics     Metrics
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │   LangGraph Pipeline    │
              │                         │
              │  ┌───────────────────┐  │
              │  │ Data Quality Agent│  │  Evidently AI
              │  │  (Drift Detection)│──┼── drift check
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │ Model Eval Agent  │  │  Sliding window
              │  │  (Performance)    │──┼── F1/accuracy
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │ Retraining Agent  │  │  PyTorch +
              │  │  (Fine-tuning)    │──┼── MLflow
              │  └────────┬──────────┘  │
              │           │             │
              │  ┌────────▼──────────┐  │
              │  │ Deployment Agent  │  │  Zero-downtime
              │  │  (Model Swap)     │──┼── model swap
              │  └───────────────────┘  │
              └─────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Grafana Dashboards    │
              │   Model Performance     │
              │   Drift Monitoring      │
              └─────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| LLM Integration | OpenAI GPT-4o-mini |
| ML Model | DistilBERT (PyTorch + HuggingFace) |
| Drift Detection | Evidently AI |
| Experiment Tracking | MLflow |
| API Layer | FastAPI |
| MCP Server | FastMCP |
| Monitoring | Prometheus + Grafana |
| Database | PostgreSQL |
| Cache | Redis |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |

## Quick Start

### With Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/yourusername/agentops.git
cd agentops

# Set up environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Start everything
docker-compose up -d

# Train the initial model
docker-compose exec app python scripts/initial_train.py

# Run the demo
docker-compose exec app python scripts/demo.py
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
make dev

# Set up environment
cp .env.example .env

# Seed data and train
make seed
make train

# Run the API
make run
```

## Demo

1. Start the stack: `docker-compose up -d`
2. Open Grafana at http://localhost:3000 (admin/admin)
3. Run `python scripts/simulate_drift.py`
4. Watch drift score climb in Grafana
5. Agent pipeline auto-triggers retraining
6. Model swaps with zero downtime
7. F1 score recovers

## API Reference

Once running, visit http://localhost:8000/docs for the interactive API docs.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predict` | POST | Classify a support ticket |
| `/api/v1/health` | GET | Service health check |
| `/api/v1/agents/status` | GET | Agent pipeline status |
| `/api/v1/agents/trigger` | POST | Manually trigger pipeline |
| `/api/v1/metrics/info` | GET | Available Prometheus metrics |
| `/metrics` | GET | Prometheus metrics endpoint |

## MCP Integration

AgentOps exposes an MCP server for AI tool integration:

```bash
# Run the MCP server
python -m src.mcp.server
```

**Available MCP Tools:**
- `check_model_status` — Get current model health and metrics
- `trigger_retraining` — Trigger the full agent pipeline
- `predict_ticket` — Classify a support ticket
- `get_pipeline_history` — View past pipeline runs

## Dashboards

- **Grafana**: http://localhost:3000 — Model performance + drift monitoring
- **MLflow**: http://localhost:5000 — Experiment tracking + model registry
- **Prometheus**: http://localhost:9090 — Raw metrics
- **API Docs**: http://localhost:8000/docs — Interactive API documentation

## Project Structure

```
agentops/
├── src/
│   ├── agents/          # 4 LangGraph agents + orchestrator
│   ├── ml/              # DistilBERT model + training pipeline
│   ├── monitoring/      # Drift detection + performance tracking
│   ├── api/             # FastAPI application + routes
│   ├── mcp/             # MCP server for AI tool integration
│   ├── storage/         # PostgreSQL + MLflow + S3
│   └── utils/           # Logging utilities
├── tests/               # Comprehensive test suite
├── scripts/             # Data seeding, training, demo
├── monitoring/          # Prometheus + Grafana configs
├── docker-compose.yml   # Full stack orchestration
└── Makefile             # Common commands
```

## Development

```bash
# Run tests
make test

# Run linter
make lint

# Clean artifacts
make clean
```
