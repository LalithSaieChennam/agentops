"""Agent 3: Automatically fine-tunes the model when needed.

Triggered by the evaluation agent's decision. Pulls new data,
fine-tunes DistilBERT, evaluates the new model, and logs to MLflow.
"""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import structlog

from src.ml.model import TicketClassifier
from src.ml.train import Trainer
from src.ml.data_processor import TicketDataProcessor
from src.monitoring.metrics_exporter import RETRAINING_RUNS
from src.agents.state import AgentState
from src.config import settings

logger = structlog.get_logger()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def retraining_agent(state: AgentState) -> AgentState:
    """Fine-tune the model on updated data.

    1. Checks if retraining is warranted (from Agent 2's decision)
    2. Loads updated training data (including recent production data)
    3. Fine-tunes DistilBERT
    4. Evaluates new model against test set
    5. Logs everything to MLflow
    6. Decides if new model is good enough to deploy
    """
    logger.info("retraining_agent_started")

    # Check if retraining is needed
    should_retrain = state.get("performance_degraded", False) or state.get("drift_detected", False)

    if not should_retrain:
        state["retraining_triggered"] = False
        state["retraining_summary"] = "No retraining needed. Model performance is within acceptable bounds."
        return state

    try:
        RETRAINING_RUNS.inc()

        # Load fresh data (includes recent production samples if labeled)
        processor = TicketDataProcessor()
        train_dataset, val_dataset, test_dataset = processor.load_and_prepare()

        # Initialize fresh model for fine-tuning
        model = TicketClassifier()

        # Load current best weights as starting point
        try:
            model.load("models/best")
            logger.info("loaded_existing_model_for_finetuning")
        except Exception:
            logger.info("starting_from_pretrained_distilbert")

        # Train
        trainer = Trainer(
            model=model,
            learning_rate=settings.fine_tune_learning_rate,
            batch_size=settings.batch_size,
            num_epochs=settings.fine_tune_epochs,
        )
        results = trainer.train(train_dataset, val_dataset, experiment_name="agentops_retraining")

        # LLM evaluates the training results
        evaluation = llm.invoke([HumanMessage(content=f"""
You are an MLOps retraining agent. Evaluate these training results:

- New model F1: {results['best_f1']:.4f}
- Previous baseline F1: {state.get('baseline_f1', 'N/A')}
- F1 improvement: {results['best_f1'] - state.get('baseline_f1', 0):.4f}

Should we deploy this new model? Consider:
- Is the new F1 better than the degraded model?
- Is it close to or better than the original baseline?
- Are there any red flags?

Respond with DEPLOY or SKIP and your reasoning.
""")])

        state["retraining_triggered"] = True
        state["new_model_f1"] = results["best_f1"]
        state["new_model_path"] = results["model_path"]
        state["retraining_summary"] = evaluation.content
        state["mlflow_run_id"] = results.get("mlflow_run_id", "")

        logger.info("retraining_agent_complete", new_f1=results["best_f1"])

    except Exception as e:
        logger.error("retraining_agent_failed", error=str(e))
        state["errors"] = state.get("errors", []) + [f"Retraining agent: {str(e)}"]
        state["retraining_triggered"] = False

    return state
