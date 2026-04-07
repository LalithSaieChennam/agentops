# AgentOps - Agentic MLOps Pipeline

[![CI](https://github.com/aabdel0181/agentops/actions/workflows/ci.yml/badge.svg)](https://github.com/aabdel0181/agentops/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)]()
[![Docker](https://img.shields.io/badge/docker-6%20containers-blue.svg)]()

**Autonomous ML model monitoring, retraining, and deployment powered by LangGraph agents.**

AgentOps orchestrates 4 autonomous agents that monitor a deployed DistilBERT classifier, detect data drift and performance degradation, automatically retrain when quality drops, and deploy updated models with zero downtime.

## Architecture

```
                    +----------------+
                    |   Incoming     |
                    |   Tickets      |
                    +-------+--------+
                            |
                    +-------v--------+
                    |    FastAPI     |---- /predict ---- Prometheus
                    |    + MCP      |---- /metrics      Metrics
                    +-------+--------+
                            |
              +-------------v--------------+
              |    LangGraph Pipeline      |
              |                            |
              |  +----------------------+  |
              |  | Data Quality Agent   |  |  Evidently AI
              |  |  (Drift Detection)   |--+-- drift check
              |  +----------+-----------+  |
              |             |              |
              |  +----------v-----------+  |
              |  | Model Eval Agent     |  |  Sliding window
              |  |  (Performance)       |--+-- F1/accuracy
              |  +----------+-----------+  |
              |             |              |
              |  +----------v-----------+  |
              |  | Retraining Agent     |  |  PyTorch +
              |  |  (Fine-tuning)       |--+-- MLflow
              |  +----------+-----------+  |
              |             |              |
              |  +----------v-----------+  |
              |  | Deployment Agent     |  |  Zero-downtime
              |  |  (Model Swap)        |--+-- model swap
              |  +----------------------+  |
              +----------------------------+
                            |
              +-------------v--------------+
              |    Grafana Dashboards      |
              |    Model Performance       |
              |    Drift Monitoring        |
              +----------------------------+
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
git clone https://github.com/aabdel0181/agentops.git
cd agentops

# Set up environment
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Start everything
docker compose up -d

# Train the initial model
docker compose exec app python scripts/initial_train.py

# Run the demo
docker compose exec app python scripts/demo.py
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

## How It Works

1. **Incoming requests** hit the FastAPI `/predict` endpoint
2. The **DistilBERT model** classifies support tickets into 5 categories (billing, technical, account, feature_request, general)
3. **Prometheus** collects prediction metrics (latency, confidence, class distribution)
4. The **Data Quality Agent** monitors for data drift using Evidently AI
5. The **Model Eval Agent** tracks F1/accuracy over a sliding window
6. If drift or degradation is detected, the **Retraining Agent** fine-tunes the model
7. The **Deployment Agent** performs a zero-downtime model swap
8. **Grafana dashboards** visualize everything in real-time

## Demo

1. Start the stack: `docker compose up -d`
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
| `/` | GET | Project overview and quick links |
| `/api/v1/predict` | POST | Classify a support ticket |
| `/api/v1/health` | GET | Service health check |
| `/api/v1/agents/status` | GET | Agent pipeline status |
| `/api/v1/agents/trigger` | POST | Manually trigger pipeline |
| `/api/v1/metrics/info` | GET | Available Prometheus metrics |
| `/metrics` | GET | Prometheus metrics endpoint |

### Example Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice on my credit card"}'
```

```json
{
  "label": "billing",
  "label_id": 0,
  "confidence": 0.989,
  "probabilities": {
    "billing": 0.989,
    "technical": 0.002,
    "account": 0.001,
    "feature_request": 0.005,
    "general": 0.003
  }
}
```

## MCP Integration

AgentOps exposes an MCP server for AI tool integration:

```bash
python -m src.mcp.server
```

**Available MCP Tools:**
- `check_model_status` - Get current model health and metrics
- `trigger_retraining` - Trigger the full agent pipeline
- `predict_ticket` - Classify a support ticket
- `get_pipeline_history` - View past pipeline runs

## Dashboards

| Dashboard | URL | Description |
|-----------|-----|-------------|
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Grafana | http://localhost:3000 | Model performance + drift monitoring |
| MLflow | http://localhost:5000 | Experiment tracking + model registry |
| Prometheus | http://localhost:9090 | Raw metrics |

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
├── tests/               # 66 tests across all modules
├── scripts/             # Data seeding, training, demo
├── monitoring/          # Prometheus + Grafana configs
├── docker-compose.yml   # Full stack orchestration (6 containers)
└── Makefile             # Common commands
```

## Development

```bash
make test        # Run all 66 tests
make lint        # Run ruff linter + mypy type checking
make train       # Train the model
make simulate    # Simulate data drift
make demo        # Run the full demo
make up          # Start Docker stack
make down        # Stop Docker stack
```

## Model Performance

| Metric | Score |
|--------|-------|
| Overall F1 | 0.89 |
| Accuracy | 0.91 |
| Billing F1 | 0.98 |
| Feature Request F1 | 0.97 |
| General F1 | 0.85 |
| Technical F1 | 0.82 |
| Account F1 | 0.78 |

## License

MIT
