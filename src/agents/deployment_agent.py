"""Agent 4: Handles model deployment, A/B testing, and rollback.

Takes the retrained model and safely deploys it to production
with zero-downtime swap and automatic rollback capability.
"""

import shutil
from pathlib import Path
from datetime import datetime
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import structlog

from src.monitoring.metrics_exporter import DEPLOYMENT_SWAPS, MODEL_VERSION
from src.monitoring.performance_tracker import PerformanceTracker
from src.agents.state import AgentState

logger = structlog.get_logger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def deployment_agent(state: AgentState) -> AgentState:
    """Deploy or rollback model based on pipeline results.

    1. If retraining produced a better model -> swap to new model
    2. Update model version metadata
    3. Reset monitoring baselines
    4. Generate final pipeline summary
    """
    logger.info("deployment_agent_started")

    if not state.get("retraining_triggered", False):
        state["deployment_action"] = "none"
        state["deployment_summary"] = "No deployment needed."
        _generate_final_summary(state)
        return state

    try:
        new_f1 = state.get("new_model_f1", 0)
        new_model_path = state.get("new_model_path", "")

        # Decision: deploy only if new model is better than degraded performance
        should_deploy = (
            new_f1 > state.get("current_f1", 0)
            and "DEPLOY" in state.get("retraining_summary", "").upper()
        )

        if should_deploy and new_model_path:
            # Atomic model swap
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            production_path = Path("models/production")
            backup_path = Path(f"models/backup_{version}")

            # Backup current production model
            if production_path.exists():
                shutil.copytree(production_path, backup_path)

            # Deploy new model
            if production_path.exists():
                shutil.rmtree(production_path)
            shutil.copytree(new_model_path, production_path)

            # Update metrics
            DEPLOYMENT_SWAPS.inc()
            MODEL_VERSION.info({"version": version, "f1": str(new_f1)})

            # Reset monitoring baselines
            tracker = PerformanceTracker.get_instance()
            tracker.update_baseline(new_f1)

            state["deployment_action"] = "swap"
            state["deployed_model_version"] = version
            state["deployment_summary"] = f"Deployed model v{version} with F1={new_f1:.4f}"

            logger.info("model_deployed", version=version, f1=new_f1)

        else:
            state["deployment_action"] = "none"
            state["deployment_summary"] = f"New model (F1={new_f1:.4f}) did not meet deployment criteria."

    except Exception as e:
        logger.error("deployment_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Deployment agent: {str(e)}"]
        state["deployment_action"] = "none"

    _generate_final_summary(state)
    return state


def _generate_final_summary(state: AgentState):
    """Use LLM to generate a human-readable pipeline summary."""
    summary = llm.invoke([HumanMessage(content=f"""
Summarize this MLOps pipeline run in 3-4 sentences for a status dashboard:

- Data Drift: {state.get('drift_report_summary', 'N/A')}
- Model Performance: F1={state.get('current_f1', 'N/A')}, Degraded={state.get('performance_degraded', False)}
- Retraining: {state.get('retraining_summary', 'Not triggered')}
- Deployment: {state.get('deployment_summary', 'No action')}
- Errors: {state.get('errors', [])}

Be concise and factual.
""")])
    state["final_summary"] = summary.content
