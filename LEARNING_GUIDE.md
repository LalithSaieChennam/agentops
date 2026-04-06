# AgentOps — Complete Learning Guide

## From Zero to Understanding Every Line

This document explains **everything** in the AgentOps project — not just what we built, but **why** we built it that way, what problem each piece solves, and how they all connect. Read this front to back and you'll be able to explain every architectural decision in an interview.

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [MLOps Fundamentals](#2-mlops-fundamentals)
3. [Why Agents? Why Not Just Scripts?](#3-why-agents-why-not-just-scripts)
4. [The ML Model — DistilBERT Deep Dive](#4-the-ml-model--distilbert-deep-dive)
5. [Data Pipeline — From Raw Text to Tensors](#5-data-pipeline--from-raw-text-to-tensors)
6. [Training Pipeline — How Fine-Tuning Works](#6-training-pipeline--how-fine-tuning-works)
7. [Drift Detection — Why Models Go Stale](#7-drift-detection--why-models-go-stale)
8. [Performance Monitoring — The Sliding Window](#8-performance-monitoring--the-sliding-window)
9. [The 4 Agents — LangGraph Architecture](#9-the-4-agents--langgraph-architecture)
10. [The Orchestrator — How LangGraph Wires It All](#10-the-orchestrator--how-langgraph-wires-it-all)
11. [FastAPI — Serving Predictions at Scale](#11-fastapi--serving-predictions-at-scale)
12. [MCP Server — The AI Integration Layer](#12-mcp-server--the-ai-integration-layer)
13. [Prometheus + Grafana — Observability](#13-prometheus--grafana--observability)
14. [PostgreSQL — Why We Log Everything](#14-postgresql--why-we-log-everything)
15. [MLflow — Experiment Tracking](#15-mlflow--experiment-tracking)
16. [Docker — Containerization Strategy](#16-docker--containerization-strategy)
17. [CI/CD — Automated Quality Gates](#17-cicd--automated-quality-gates)
18. [Design Patterns Used](#18-design-patterns-used)
19. [How the Full Demo Works End-to-End](#19-how-the-full-demo-works-end-to-end)
20. [Common Interview Questions](#20-common-interview-questions)

---

## 1. The Problem We're Solving

### The Real-World Pain

Imagine you work at a company that handles customer support. You've trained a machine learning model to automatically classify incoming tickets into categories: billing, technical, account, feature_request, general. When a ticket comes in saying "I was charged twice," the model tags it as "billing" and routes it to the billing team.

**Day 1:** Your model works great. 90% accuracy. Everyone's happy.

**Month 3:** Your company launches a new product. Suddenly, 60% of tickets are about technical issues with the new product. But your model was trained when technical tickets were only 25% of the total. The model starts misclassifying things. Billing issues get routed to the tech team. Feature requests get lost. Customer satisfaction drops.

**This is called "model drift"** — when the real-world data distribution changes and your model's training data no longer represents reality.

### What Companies Do Today (The Manual Way)

1. An engineer notices metrics declining (often too late)
2. They manually pull new data
3. They retrain the model on their laptop
4. They evaluate it
5. They deploy it (often with downtime)
6. Repeat in 3 months when it breaks again

### What AgentOps Does (The Autonomous Way)

AgentOps automates this entire cycle with 4 AI agents:

1. **Data Quality Agent** continuously monitors for drift
2. **Model Evaluation Agent** tracks if predictions are getting worse
3. **Retraining Agent** automatically fine-tunes the model
4. **Deployment Agent** safely swaps to the new model

No human intervention needed. The system heals itself.

---

## 2. MLOps Fundamentals

### What is MLOps?

MLOps = Machine Learning + Operations. It's the practice of deploying and maintaining ML models **in production** reliably.

Think of it like DevOps for ML:
- **DevOps** = writing code → testing → deploying → monitoring → fixing
- **MLOps** = training models → evaluating → deploying → monitoring for drift → retraining

### The ML Lifecycle

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Train   │───▶│ Evaluate │───▶│  Deploy  │───▶│ Monitor  │
└─────────┘    └──────────┘    └──────────┘    └────┬─────┘
     ▲                                              │
     │                                              │
     └──────────── Retrain when degraded ◀──────────┘
```

This is a **continuous loop**, not a one-time process. AgentOps automates this loop.

### Key MLOps Concepts in This Project

| Concept | What It Means | Where We Use It |
|---------|--------------|-----------------|
| **Model Registry** | Version control for models | MLflow |
| **Experiment Tracking** | Log every training run's metrics | MLflow |
| **Data Drift** | Input data distribution changes | Evidently AI |
| **Concept Drift** | The relationship between inputs and outputs changes | Performance Tracker |
| **Model Serving** | Making predictions available via API | FastAPI |
| **A/B Testing** | Comparing old vs new model | Deployment Agent |
| **Rollback** | Going back to previous model if new one is worse | Deployment Agent |
| **Observability** | Knowing what's happening in real-time | Prometheus + Grafana |

---

## 3. Why Agents? Why Not Just Scripts?

### The Script Approach

You could write a simple Python script:

```python
# pseudo-code
if drift_detected():
    retrain_model()
    deploy_model()
```

**Problems with scripts:**
- **Rigid logic** — can't reason about edge cases
- **No context** — doesn't consider drift + performance together
- **No explanation** — doesn't tell you WHY it made a decision
- **Hard to extend** — adding new logic means rewriting everything

### The Agent Approach

Agents use an LLM (GPT-4o-mini) to **reason** about what to do. They look at the data, consider the context, and make nuanced decisions.

For example, the Model Evaluation Agent doesn't just check "is F1 below threshold." It considers:
- How much did F1 drop?
- Is drift also happening?
- Is the confidence dropping?
- Should we retrain urgently or just monitor?

This is closer to how a human ML engineer thinks.

### Why LangGraph Specifically?

LangGraph is a framework for building **stateful, multi-agent workflows**. Here's why we chose it:

1. **State management** — Agents share a state dict. Each agent reads from it and writes to it. No complex message passing.
2. **Conditional routing** — We can say "if drift detected, go to retraining; otherwise skip to deployment."
3. **Graph-based** — The pipeline is a directed graph, which is easy to visualize and reason about.
4. **Async support** — Agents can run concurrently when independent.
5. **Built on LangChain** — Uses the same ecosystem for LLM calls.

---

## 4. The ML Model — DistilBERT Deep Dive

### What is BERT?

BERT (Bidirectional Encoder Representations from Transformers) is a language model created by Google in 2018. It revolutionized NLP because it understands **context** — the word "bank" means something different in "river bank" vs "bank account."

### What is DistilBERT?

DistilBERT is a **smaller, faster version** of BERT. It was created by Hugging Face using a technique called **knowledge distillation**:

- **BERT**: 110 million parameters, 12 layers
- **DistilBERT**: 66 million parameters, 6 layers
- **Speed**: DistilBERT is 60% faster
- **Accuracy**: Retains 97% of BERT's performance

**Why we chose DistilBERT:**
- Fast enough to fine-tune on a laptop (no GPU required, though GPU helps)
- Small enough to deploy in a Docker container
- Accurate enough for text classification
- Well-documented and widely used

### How Text Classification Works

```
Input: "I was charged twice for my subscription"
                    │
                    ▼
         ┌──────────────────┐
         │   Tokenizer      │  "I" → 1045, "was" → 2001, ...
         │   (text → numbers)│
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │   DistilBERT     │  Understands meaning & context
         │   (6 transformer │  Outputs 768-dim vector per token
         │    layers)        │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │   Classification │  768-dim → 5 classes
         │   Head (linear)  │  [0.01, 0.02, 0.95, 0.01, 0.01]
         └────────┬─────────┘
                  │
                  ▼
         Output: "billing" (confidence: 0.95)
```

### The Code: `src/ml/model.py`

```python
class TicketClassifier:
    def __init__(self, model_name, num_labels, device):
        # Load pre-trained DistilBERT and add a classification head
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
```

**What `from_pretrained` does:**
1. Downloads the pre-trained DistilBERT weights (if not cached)
2. These weights were trained on massive text corpora (Wikipedia, BookCorpus)
3. The model already "understands" English
4. We just need to teach it our specific classification task

**What `num_labels=5` does:**
- Adds a **classification head** (a linear layer) on top of DistilBERT
- This layer maps the 768-dimensional output to 5 classes
- This head has random weights initially — training teaches it our categories

### Inference (Prediction)

```python
def predict(self, input_ids, attention_mask):
    self.model.eval()           # Turn off dropout (training randomness)
    with torch.no_grad():       # Don't compute gradients (saves memory)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(outputs.logits, dim=-1)  # Convert to probabilities
        confidence, predicted = torch.max(probabilities, dim=-1)  # Get highest prob
```

**Key concepts:**
- `model.eval()` — Tells PyTorch we're inferring, not training. Disables dropout and batch normalization.
- `torch.no_grad()` — Disables gradient computation. During inference, we don't need gradients (those are only for training). This saves ~50% memory and is faster.
- `F.softmax` — Converts raw model outputs (logits) into probabilities that sum to 1.0.
- `attention_mask` — Tells the model which tokens are real text vs padding. Since we pad all inputs to the same length, we need to tell the model "ignore the padding tokens."

---

## 5. Data Pipeline — From Raw Text to Tensors

### Why We Need a Data Pipeline

ML models don't understand text. They understand numbers (tensors). The data pipeline converts:

```
"I was charged twice" → [101, 1045, 2001, 5765, 3807, 102, 0, 0, ...]
```

### The Code: `src/ml/data_processor.py`

#### Step 1: Label Mapping

```python
LABEL_MAP = {
    "billing": 0,
    "technical": 1,
    "account": 2,
    "feature_request": 3,
    "general": 4,
}
```

**Why numbers?** Neural networks do math. They can't do math on the word "billing." So we map each category to a number. The reverse mapping (`LABEL_NAMES`) converts back for human-readable output.

#### Step 2: Tokenization

```python
self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
```

**What a tokenizer does:**
1. Splits text into subwords: "charging" → ["charg", "##ing"]
2. Maps subwords to IDs: ["charg", "##ing"] → [14878, 2075]
3. Adds special tokens: [CLS] at start, [SEP] at end
4. Pads to a fixed length (128 tokens)

**Why "uncased"?** Means lowercase everything. "Billing" and "billing" become the same. Simpler and works well for classification.

**Why subword tokenization?** It handles words the model has never seen. Even if "misclassifying" isn't in the vocabulary, "mis", "##class", "##ify", "##ing" are.

#### Step 3: Dataset Splits

```python
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
```

**Why three splits?**
- **Training set (70%)** — The model learns from this
- **Validation set (15%)** — Used during training to check progress (prevents overfitting)
- **Test set (15%)** — Final evaluation after training is done. Never seen during training.

**Why `stratify`?** Ensures each split has the same proportion of each class. Without it, your test set might accidentally have 50% billing and 5% technical, giving misleading metrics.

**Why `random_state=42`?** Makes the split reproducible. Same seed = same split every time. Important for experiment reproducibility.

#### Step 4: HuggingFace Dataset Format

```python
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
```

This tells HuggingFace to return PyTorch tensors instead of Python lists. PyTorch tensors can be moved to GPU and processed in batches efficiently.

---

## 6. Training Pipeline — How Fine-Tuning Works

### What is Fine-Tuning?

DistilBERT was pre-trained on Wikipedia and BookCorpus. It already understands English. **Fine-tuning** means taking this pre-trained model and training it a little more on our specific task (ticket classification).

Think of it like this:
- **Pre-training** = Going to medical school (general knowledge)
- **Fine-tuning** = Doing a residency in cardiology (specialized knowledge)

### The Training Loop: `src/ml/train.py`

```python
for epoch in range(self.num_epochs):
    self.model.model.train()  # Enable training mode
    for batch in train_loader:
        optimizer.zero_grad()     # 1. Reset gradients
        outputs = self.model.model(...)  # 2. Forward pass
        loss = outputs.loss       # 3. Compute loss
        loss.backward()           # 4. Backward pass (compute gradients)
        torch.nn.utils.clip_grad_norm_(...)  # 5. Clip gradients
        optimizer.step()          # 6. Update weights
        scheduler.step()          # 7. Update learning rate
```

Let's break each step down:

#### Step 1: `optimizer.zero_grad()`
PyTorch **accumulates** gradients by default. If you don't zero them, the gradients from the previous batch add to the current batch's gradients. Always reset before each batch.

#### Step 2: Forward Pass
Feed the batch through the model. The model outputs:
- `logits` — Raw scores for each class (before softmax)
- `loss` — Cross-entropy loss (how wrong the predictions are)

#### Step 3: Loss Function
**Cross-entropy loss** measures how far the predicted probability distribution is from the actual labels. If the model says [0.1, 0.8, 0.05, 0.03, 0.02] but the true label is class 0, the loss is high. If it correctly says [0.95, 0.02, 0.01, 0.01, 0.01], the loss is low.

#### Step 4: `loss.backward()`
**Backpropagation** — the key algorithm of deep learning. It computes the **gradient** of the loss with respect to every parameter in the model. Gradients tell us: "if I change this parameter a tiny bit, how does the loss change?"

#### Step 5: Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
```
Sometimes gradients get really large (the "exploding gradients" problem). This caps the gradient magnitude at 1.0, preventing the model from making huge, unstable updates.

#### Step 6: `optimizer.step()`
**AdamW optimizer** updates every parameter using its gradient:
```
new_weight = old_weight - learning_rate * gradient
```
AdamW is an improved version of Adam that handles weight decay (regularization) correctly. It's the standard optimizer for fine-tuning transformers.

#### Step 7: Learning Rate Scheduler
```python
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=total_steps
)
```

**Warmup:** Start with a tiny learning rate and gradually increase it over the first 100 steps. This prevents the model from making big, chaotic updates early in training when the gradients are noisy.

**Linear decay:** After warmup, gradually decrease the learning rate to zero. This helps the model "settle" into a good solution instead of bouncing around.

```
Learning Rate
     ▲
     │   /\
     │  /  \
     │ /    \
     │/      \
     └────────────▶ Steps
      warmup  decay
```

### Hyperparameters Explained

| Parameter | Our Value | Why |
|-----------|----------|-----|
| `learning_rate` | 2e-5 | Standard for BERT fine-tuning. Too high = model forgets pre-trained knowledge. Too low = doesn't learn. |
| `batch_size` | 16 | Fits in memory. Larger = more stable gradients but needs more memory. |
| `num_epochs` | 3 | DistilBERT typically converges in 2-4 epochs for classification. More = risk of overfitting. |
| `warmup_steps` | 100 | ~6% of total steps. Prevents early instability. |
| `weight_decay` | 0.01 | Regularization to prevent overfitting. Penalizes large weights. |

### MLflow Integration

```python
with mlflow.start_run():
    mlflow.log_params({...})        # Log what hyperparameters we used
    mlflow.log_metric("val_f1", ...) # Log metrics at each epoch
    mlflow.log_artifact("models/best") # Save the model files
```

**Why MLflow?**
- You train the model 50 times with different hyperparameters
- Without tracking, you lose track of which run produced the best model
- MLflow logs everything: parameters, metrics, artifacts
- You can compare runs in the MLflow UI (http://localhost:5000)

---

## 7. Drift Detection — Why Models Go Stale

### Types of Drift

#### Data Drift (Covariate Shift)
The **input data distribution changes**, but the relationship between inputs and outputs stays the same.

**Example:** Your model was trained when 25% of tickets were about billing. Now, after a price increase, 60% are about billing. The model hasn't seen this distribution before and starts making mistakes.

#### Concept Drift
The **relationship between inputs and outputs changes**. The same text now means something different.

**Example:** "I can't access my account" used to mean "password reset" (account category). But after launching a new feature called "Access Control," it now means "feature isn't working" (technical category).

#### Prediction Drift
The **model's predictions** change distribution, even if the input data looks the same.

### How Evidently AI Detects Drift

The code is in `src/monitoring/drift_detector.py`.

Evidently uses **statistical tests** to compare two datasets:
1. **Reference data** — Your training data (the "known good" distribution)
2. **Current data** — Recent production data

For numerical features, it uses the **Kolmogorov-Smirnov test**:
- Computes the maximum difference between two cumulative distribution functions
- If the difference exceeds a threshold, drift is detected

For categorical features, it uses **Jensen-Shannon divergence**:
- Measures the similarity between two probability distributions
- Returns a value between 0 (identical) and 1 (completely different)

```python
report = Report(metrics=[
    DatasetDriftMetric(),  # Overall: is the dataset drifted?
    DataDriftTable(),      # Per-column: which features drifted?
])
report.run(reference_data=self.reference_data, current_data=current_data)
```

### The DriftReport Dataclass

```python
@dataclass
class DriftReport:
    is_drifted: bool           # Did the dataset drift overall?
    drift_score: float         # What fraction of columns drifted? (0.0 to 1.0)
    drifted_columns: list      # Which specific columns drifted?
    column_scores: dict        # Per-column drift scores
    details: dict              # Full Evidently report (for debugging)
```

**Why a dataclass?** It's a clean, typed container for drift results. Better than returning a raw dict because your IDE can autocomplete fields and you catch typos at import time, not runtime.

### The Singleton Pattern

```python
class DriftDetector:
    _instance = None

    @classmethod
    def get_instance(cls, reference_data=None):
        if cls._instance is None:
            cls._instance = cls(reference_data)
        return cls._instance
```

**Why singleton?** The DriftDetector holds the reference data in memory. We don't want to reload it every time an agent runs. The singleton ensures one instance exists throughout the application lifecycle. All agents share the same detector.

---

## 8. Performance Monitoring — The Sliding Window

### The Code: `src/monitoring/performance_tracker.py`

### What is a Sliding Window?

Instead of looking at ALL predictions ever made, we look at the last N predictions (default: 500). This is called a **sliding window**.

```
Time →
[....................|=====window=====|]
 older predictions    last 500 preds
 (ignored)            (analyzed)
```

**Why sliding window?**
- Recent data is more relevant than old data
- If the model was bad 3 months ago but good now, we don't want old predictions pulling down the metrics
- Keeps memory usage constant

```python
self.predictions = deque(maxlen=window_size)
```

`deque(maxlen=500)` is a Python data structure that automatically drops the oldest element when you add a new one past the max size. Exactly what we need.

### Metrics We Track

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Accuracy** | % of correct predictions | Simple but misleading if classes are imbalanced |
| **F1 Score (weighted)** | Harmonic mean of precision and recall | Our primary metric — handles class imbalance |
| **Precision** | Of tickets labeled "billing," how many actually were? | Important for routing accuracy |
| **Recall** | Of actual billing tickets, how many did we catch? | Important for not missing tickets |
| **Confidence Mean** | Average model confidence | Dropping confidence often precedes accuracy drops |

### Why F1 and Not Just Accuracy?

Imagine 90% of tickets are "billing." A model that always predicts "billing" gets 90% accuracy but is useless — it never identifies technical issues.

**F1 score** penalizes this:
- **Precision** for "technical" = 0% (never predicted it)
- **Recall** for "technical" = 0% (missed all technical tickets)
- **F1** for "technical" = 0%

Weighted F1 accounts for class sizes, so classes with more samples contribute more.

### Degradation Detection

```python
def is_degraded(self):
    f1_drop = self.baseline_f1 - snapshot.f1_weighted
    is_degraded = f1_drop > self.degradation_threshold  # 5% drop
```

**How it works:**
1. We set a baseline F1 when the model is first deployed (e.g., 0.88)
2. We continuously compute F1 over the sliding window
3. If F1 drops more than 5% below baseline (0.88 → 0.83), we flag degradation
4. The agents then decide what to do about it

**Why 5%?** It's a common threshold in industry. Small fluctuations are normal (maybe one bad batch of predictions). A sustained 5% drop indicates a real problem.

---

## 9. The 4 Agents — LangGraph Architecture

### Agent State: The Shared Memory

All agents communicate through a shared state object (`src/agents/state.py`):

```python
class AgentState(TypedDict):
    # Each agent writes its section
    drift_detected: bool        # Agent 1 writes this
    performance_degraded: bool  # Agent 2 writes this
    retraining_triggered: bool  # Agent 3 writes this
    deployment_action: str      # Agent 4 writes this
```

**Why TypedDict?** It's a Python dict with type hints. LangGraph requires it for state management. The type hints let your IDE catch mistakes and make the code self-documenting.

### Agent 1: Data Quality Agent

**File:** `src/agents/data_quality_agent.py`

**What it does:**
1. Pulls last 1000 predictions from PostgreSQL
2. Runs Evidently drift detection against the training reference data
3. Asks GPT-4o-mini to interpret the results
4. Updates Prometheus metrics
5. Writes drift results to the shared state

**Why use an LLM here?** The raw drift metrics (e.g., "drift_score: 0.42, drifted_columns: ['confidence', 'predicted_label_id']") aren't very useful to a human. The LLM translates this into: "Moderate drift detected. The distribution of technical tickets has increased significantly, possibly due to a product launch. This is concerning and may require model retraining."

This summary goes into Grafana dashboards and pipeline reports.

### Agent 2: Model Evaluation Agent

**File:** `src/agents/model_eval_agent.py`

**What it does:**
1. Computes sliding window metrics (F1, accuracy, etc.)
2. Checks if performance has degraded
3. Cross-references with Agent 1's drift findings
4. Uses LLM to make a nuanced decision

**The decision matrix:**
```
Drift? │ Degraded? │ Decision
───────┼───────────┼──────────
  Yes  │    Yes    │ RETRAIN (urgent — drift + performance drop)
  No   │    Yes    │ RETRAIN (model issue, not data issue)
  Yes  │    No     │ MONITOR (drift hasn't impacted performance yet)
  No   │    No     │ NO ACTION (everything's fine)
```

**Why not just use the decision matrix directly?** The LLM adds nuance. Maybe F1 dropped 4.8% (just under the 5% threshold) AND drift is detected. A rigid script would say "no action" but the LLM might recommend retraining because the combination is concerning.

### Agent 3: Retraining Agent

**File:** `src/agents/retraining_agent.py`

**What it does:**
1. Checks if retraining is needed (from Agent 2's state)
2. Loads fresh training data
3. Fine-tunes DistilBERT with a lower learning rate (1e-5 vs 2e-5)
4. Evaluates the new model
5. Asks LLM: "Should we deploy this?"
6. Logs everything to MLflow

**Why lower learning rate for fine-tuning?**
- Initial training: 2e-5 (learning a lot from scratch)
- Re-fine-tuning: 1e-5 (small adjustments to existing knowledge)
- Too high = the model "forgets" what it learned before (catastrophic forgetting)

**Why fewer epochs (2 vs 3)?**
For re-fine-tuning, the model already has a good starting point. It only needs small adjustments. More epochs risk overfitting to the new data distribution.

### Agent 4: Deployment Agent

**File:** `src/agents/deployment_agent.py`

**What it does:**
1. Decides if the new model is good enough to deploy
2. Backs up the current production model
3. Performs an atomic model swap
4. Updates Prometheus metrics
5. Resets the performance baseline
6. Generates a final pipeline summary

**Atomic model swap:**
```python
# 1. Backup current model
shutil.copytree(production_path, backup_path)

# 2. Remove old production model
shutil.rmtree(production_path)

# 3. Copy new model to production
shutil.copytree(new_model_path, production_path)
```

**Why backup?** If the new model performs poorly in production, we can rollback to the backup. This is a safety net.

**Why "atomic"?** The model swap happens in two steps (remove old + copy new). While not truly atomic (a crash between steps would leave no production model), it's good enough for a single-server setup. In production at scale, you'd use blue-green deployment or a load balancer.

---

## 10. The Orchestrator — How LangGraph Wires It All

### The Code: `src/agents/orchestrator.py`

### The Pipeline Graph

```
┌───────────────┐
│  data_quality  │  (always runs first)
└───────┬───────┘
        │
        ▼
┌───────────────┐
│  model_eval    │  (always runs second)
└───────┬───────┘
        │
        ├── if degraded or drifted ──▶ ┌─────────┐
        │                               │ retrain  │
        │                               └────┬────┘
        │                                    │
        ▼                                    ▼
┌───────────────┐◀──────────────────────────┘
│    deploy      │  (always runs last)
└───────┬───────┘
        │
        ▼
      [END]
```

### Key LangGraph Concepts

#### StateGraph
```python
workflow = StateGraph(AgentState)
```
Creates a graph where the state is passed between nodes. Each node (agent function) receives the state, modifies it, and returns it.

#### Nodes
```python
workflow.add_node("data_quality", data_quality_agent)
```
Each node is a Python function that takes `AgentState` and returns `AgentState`.

#### Edges
```python
workflow.add_edge("data_quality", "model_eval")  # Always go from DQ to eval
```
Unconditional edge: always go from A to B.

#### Conditional Edges
```python
workflow.add_conditional_edges("model_eval", should_retrain, {
    "retrain": "retrain",
    "deploy": "deploy",
})
```
The `should_retrain` function looks at the state and returns either "retrain" or "deploy". LangGraph follows the corresponding edge.

#### Compilation
```python
pipeline = workflow.compile()
```
Converts the graph definition into an executable pipeline. After this, you can call `pipeline.ainvoke(initial_state)`.

#### Async Invocation
```python
result = await pipeline.ainvoke(initial_state)
```
Runs the pipeline asynchronously. Each agent runs in sequence (data_quality → model_eval → retrain/deploy). The `await` keyword means the calling code can do other things while waiting.

---

## 11. FastAPI — Serving Predictions at Scale

### The Code: `src/api/app.py`

### Why FastAPI?

| Feature | Why It Matters |
|---------|---------------|
| **Async support** | Handle many concurrent requests without blocking |
| **Auto-generated docs** | /docs endpoint gives you interactive API documentation for free |
| **Pydantic validation** | Input/output schemas are validated automatically |
| **Type hints** | Catch errors at development time, not production |
| **Performance** | One of the fastest Python web frameworks |

### Lifespan Events

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model into memory
    global model, processor
    model = TicketClassifier()
    model.load("models/production")
    processor = TicketDataProcessor()
    yield  # App runs here
    # Shutdown: cleanup
    logger.info("shutting_down")
```

**Why lifespan?** Loading a DistilBERT model takes a few seconds and significant memory. We load it **once** at startup and keep it in memory. Every prediction request reuses the same model instance.

**Why `global`?** The model needs to be accessible from route handlers. In a larger app, you'd use dependency injection, but globals work fine for this scope.

### The Prediction Endpoint

```python
@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
```

**Request flow:**
1. Client sends POST with `{"text": "I was charged twice"}`
2. Pydantic validates the input (must have a `text` field)
3. Text is tokenized
4. Model runs inference
5. Prometheus metrics are updated
6. Prediction is logged to PostgreSQL
7. Response is returned with label, confidence, and probabilities

**Why log predictions to the database?** This is the data the drift detector uses. Without logged predictions, we can't detect drift.

### Background Tasks

```python
@router.post("/agents/trigger")
async def trigger_pipeline(background_tasks: BackgroundTasks):
    background_tasks.add_task(_run_and_store, "manual")
    return PipelineTriggerResponse(message="Pipeline triggered", ...)
```

**Why background?** The agent pipeline can take minutes (especially retraining). We don't want the HTTP request to time out. `BackgroundTasks` tells FastAPI to run the function after sending the response. The client gets an immediate "Pipeline triggered" response and can poll the status endpoint.

---

## 12. MCP Server — The AI Integration Layer

### What is MCP?

MCP (Model Context Protocol) is an open protocol created by Anthropic that lets AI tools (Claude, Cursor, etc.) interact with external systems. Think of it as "USB for AI" — a standard way for AI assistants to use tools.

### The Code: `src/mcp/server.py`

### How It Works

```python
from fastmcp import FastMCP

mcp = FastMCP("AgentOps", description="Agentic MLOps Pipeline")

@mcp.tool()
async def check_model_status() -> dict:
    """Check the current health and performance of the deployed model."""
    ...
```

When you connect Claude or Cursor to this MCP server, the AI can:
- Ask "How's my model doing?" → calls `check_model_status`
- Say "Retrain the model" → calls `trigger_retraining`
- Say "Classify this ticket: I was charged twice" → calls `predict_ticket`

### Why MCP?

This is a **differentiator** for your resume. It shows you understand:
1. AI tool integration (not just building models, but making them accessible)
2. Protocol design (exposing functions as tools)
3. The MCP ecosystem (growing standard in the AI industry)

### Tools Exposed

| Tool | What It Does | When AI Would Use It |
|------|-------------|---------------------|
| `check_model_status` | Returns F1, drift status, health | "How's my model doing?" |
| `trigger_retraining` | Runs the full agent pipeline | "The model seems off, retrain it" |
| `predict_ticket` | Classifies a single ticket | "What category is this ticket?" |
| `get_pipeline_history` | Lists recent pipeline runs | "What happened in the last retraining?" |

---

## 13. Prometheus + Grafana — Observability

### Why Observability?

Your model is running in production. How do you know if it's working? You need:
- **Metrics** — Numbers that describe system behavior (F1 score, latency, throughput)
- **Dashboards** — Visual representation of metrics over time
- **Alerts** — Notifications when something goes wrong

### Prometheus — The Metrics Collector

Prometheus **scrapes** metrics from your app every 15 seconds.

```yaml
# monitoring/prometheus.yml
scrape_configs:
  - job_name: "agentops"
    static_configs:
      - targets: ["app:8000"]
    metrics_path: "/metrics"
```

### Types of Prometheus Metrics

The code is in `src/monitoring/metrics_exporter.py`:

#### Counter
```python
PREDICTION_COUNT = Counter("agentops_predictions_total", "Total predictions made", ["predicted_class"])
```
**What:** A value that only goes up (never decreases).
**Use case:** Total predictions, total errors, total retraining runs.
**How to read:** "We've made 15,234 predictions since the app started."

#### Gauge
```python
MODEL_F1_SCORE = Gauge("agentops_model_f1_score", "Current model F1 score")
```
**What:** A value that can go up or down.
**Use case:** Current F1 score, current drift score, memory usage.
**How to read:** "The F1 score right now is 0.87."

#### Histogram
```python
PREDICTION_LATENCY = Histogram("agentops_prediction_latency_seconds", "Prediction latency", buckets=[...])
```
**What:** Tracks the distribution of values across predefined buckets.
**Use case:** Latency (how many requests took 0-10ms, 10-50ms, 50-100ms, etc.)
**How to read:** "95% of predictions complete in under 50ms."

#### Info
```python
MODEL_VERSION = Info("agentops_model_version", "Current deployed model version")
```
**What:** Key-value pairs for metadata.
**Use case:** Current model version, build info.

### Grafana — The Dashboard

We provide two pre-built dashboards:

**Dashboard 1: Model Performance**
- F1 Score over time (line chart) — Are we getting better or worse?
- Prediction confidence distribution (histogram) — Is the model confident or guessing?
- Predictions per class (bar chart) — Has the class distribution changed?
- Latency percentiles (line chart) — Are predictions getting slower?

**Dashboard 2: Agent Pipeline & Drift**
- Drift score over time (with threshold line) — When did drift start?
- Drift detected alert (red/green status) — At-a-glance drift status
- Pipeline runs count — How many times have agents run?
- Retraining & deployment counts — How many models were deployed?

### How Metrics Flow

```
App (FastAPI)
    │
    ├── On each prediction:
    │   ├── PREDICTION_COUNT.labels(predicted_class="billing").inc()
    │   ├── PREDICTION_CONFIDENCE.observe(0.95)
    │   └── PREDICTION_LATENCY.observe(0.023)
    │
    ├── On drift check:
    │   ├── DATA_DRIFT_SCORE.set(0.42)
    │   └── DATA_DRIFT_DETECTED.set(1)
    │
    └── Exposes all metrics at /metrics endpoint
            │
            ▼
      Prometheus (scrapes /metrics every 15s)
            │
            ▼
      Grafana (queries Prometheus, renders dashboards)
```

---

## 14. PostgreSQL — Why We Log Everything

### The Code: `src/storage/database.py`

### Two Tables

#### `predictions` — Every single prediction
```sql
id, timestamp, input_text, predicted_label, confidence, probabilities, actual_label, model_version
```

**Why log every prediction?**
1. **Drift detection** — We need recent predictions to compare against training data
2. **Performance monitoring** — When we get ground truth labels (user feedback), we can compute actual accuracy
3. **Debugging** — If something goes wrong, we can trace exactly what happened
4. **Compliance** — Many industries require prediction audit trails

#### `pipeline_runs` — Every agent pipeline execution
```sql
id, timestamp, trigger_reason, drift_detected, drift_score, performance_degraded, retraining_triggered, deployment_action, summary, duration
```

**Why log pipeline runs?**
1. **Audit trail** — "What did the agents do at 3am?"
2. **MCP queries** — The `get_pipeline_history` tool reads from this table
3. **Debugging** — If a bad model got deployed, trace back to why

### SQLAlchemy ORM

```python
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ...
```

**Why SQLAlchemy?** It's an ORM (Object-Relational Mapper). Instead of writing raw SQL, you interact with Python objects. Benefits:
- SQL injection protection (parameterized queries by default)
- Database-agnostic (switch from PostgreSQL to MySQL by changing the URL)
- Type safety
- Migration support (with Alembic)

### Session Management

```python
with SessionLocal() as session:
    session.add(prediction)
    session.commit()
```

**Why `with`?** Context manager ensures the session is closed after use, even if an error occurs. Prevents connection leaks.

---

## 15. MLflow — Experiment Tracking

### Why MLflow?

When you train a model, you make dozens of choices:
- Learning rate: 1e-5 or 2e-5?
- Batch size: 8 or 16?
- Epochs: 2 or 3?

Each combination produces a different model with different metrics. **MLflow tracks all of this.**

### What Gets Logged

```python
with mlflow.start_run():
    # Parameters (inputs to training)
    mlflow.log_params({"learning_rate": 2e-5, "batch_size": 16, ...})

    # Metrics (outputs of training)
    mlflow.log_metric("val_f1", 0.88, step=0)  # F1 at epoch 0
    mlflow.log_metric("val_f1", 0.91, step=1)  # F1 at epoch 1

    # Artifacts (files produced)
    mlflow.log_artifact("models/best")  # The actual model files
```

### The MLflow UI

At http://localhost:5000 you can:
- Compare runs side-by-side
- See which hyperparameters produced the best F1
- Download model artifacts
- Register models for deployment

### Model Registry

```python
class ModelRegistry:
    def register_model(self, run_id, model_path):
        mlflow.register_model(f"runs:/{run_id}/{model_path}", "ticket-classifier")

    def promote_to_production(self, version):
        self.client.transition_model_version_stage(
            name="ticket-classifier", version=version, stage="Production"
        )
```

The model registry is like Git for models:
- Version 1: F1 = 0.85 (initial training)
- Version 2: F1 = 0.88 (after first retraining)
- Version 3: F1 = 0.91 (after second retraining)

You can promote versions to "Production" and rollback if needed.

---

## 16. Docker — Containerization Strategy

### Why Docker?

"It works on my machine" is the #1 problem in software deployment. Docker ensures your app runs exactly the same everywhere: your laptop, your colleague's laptop, production servers.

### The Dockerfile

```dockerfile
FROM python:3.11-slim      # Start from a minimal Python image
WORKDIR /app               # Set working directory
COPY pyproject.toml .      # Copy dependency file first
RUN pip install -e ".[dev]" # Install dependencies (cached if unchanged)
COPY . .                   # Copy application code
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why copy pyproject.toml first?** Docker caches each layer. If your code changes but dependencies don't, Docker reuses the cached dependency layer. This makes rebuilds much faster (seconds instead of minutes).

### Docker Compose — The Full Stack

`docker-compose.yml` orchestrates 6 services:

| Service | Port | Purpose |
|---------|------|---------|
| **app** | 8000 | Your FastAPI application |
| **postgres** | 5432 | Prediction logs, pipeline history |
| **redis** | 6379 | Caching (model, predictions) |
| **mlflow** | 5000 | Experiment tracking UI |
| **prometheus** | 9090 | Metrics collection |
| **grafana** | 3000 | Dashboards |

**`depends_on`** ensures services start in the right order (PostgreSQL before the app).

**Volumes:**
- `pgdata` — PostgreSQL data persists between restarts
- `mlflow_data` — MLflow experiments persist
- `./models:/app/models` — Model files are mounted from your host, so they persist

---

## 17. CI/CD — Automated Quality Gates

### The Code: `.github/workflows/ci.yml`

### What CI/CD Means

- **CI (Continuous Integration)** — Automatically run tests and linting on every push
- **CD (Continuous Deployment)** — Automatically deploy when tests pass

### Our CI Pipeline

On every push to `main` or pull request:

```
1. Checkout code
2. Setup Python 3.11
3. Install dependencies
4. Run ruff (linter) — catches style issues
5. Run mypy (type checker) — catches type errors
6. Run pytest (tests) — catches logic errors
7. Upload coverage report
```

**Why lint?** Consistent code style across the team. Ruff catches things like unused imports, incorrect formatting, and potential bugs.

**Why type check?** mypy verifies that your type hints are correct. If a function says it returns `float` but actually returns `str`, mypy catches it before runtime.

**Why test?** Automated tests catch regressions. If you change the drift detector and accidentally break the orchestrator, tests catch it before it reaches production.

### Our CD Pipeline

On git tags matching `v*` (e.g., `v1.0.0`):

```
1. Build Docker image
2. Push to GitHub Container Registry (ghcr.io)
```

This means deploying is as simple as:
```bash
git tag v1.0.0
git push --tags
```

---

## 18. Design Patterns Used

### Singleton Pattern

**Where:** DriftDetector, PerformanceTracker

**Why:** These objects hold state (reference data, prediction history) that should persist throughout the application lifecycle. Creating a new instance every time would lose the accumulated data.

```python
@classmethod
def get_instance(cls):
    if cls._instance is None:
        cls._instance = cls(...)
    return cls._instance
```

### Repository Pattern

**Where:** `src/storage/database.py`

**Why:** Separates database logic from business logic. The agents call `get_recent_predictions()` — they don't need to know it's PostgreSQL. You could swap to MongoDB by changing one file.

### Pipeline Pattern

**Where:** LangGraph orchestrator

**Why:** Each agent is a stage in a pipeline. Data flows through stages sequentially. Stages can be added, removed, or reordered without changing other stages.

### Observer Pattern

**Where:** Prometheus metrics

**Why:** Various parts of the code emit metrics (predictions, drift checks, pipeline runs). Prometheus observes these metrics independently. The metric emitters don't know or care about Prometheus.

### Factory Pattern

**Where:** `build_pipeline()` in orchestrator

**Why:** Constructs the LangGraph pipeline once and returns a compiled, reusable object. The construction logic is encapsulated in one function.

---

## 19. How the Full Demo Works End-to-End

### Step-by-Step Walkthrough

```
1. docker-compose up -d
   → Starts all 6 services (app, postgres, redis, mlflow, prometheus, grafana)

2. python scripts/initial_train.py
   → Downloads DistilBERT weights
   → Downloads ag_news dataset (10,000 samples)
   → Fine-tunes for 3 epochs
   → Saves model to models/production/
   → Logs everything to MLflow

3. python scripts/simulate_drift.py
   → Phase 1: Sends 500 "normal" tickets (even distribution across categories)
      Each ticket → POST /api/v1/predict
      Each prediction → logged to PostgreSQL
      Each prediction → updates Prometheus metrics

   → Phase 2: Sends 500 "drifted" tickets (55% technical, 10% billing)
      The model starts seeing way more technical tickets than it was trained on
      Drift score climbs in Prometheus/Grafana

   → Phase 3: Triggers agent pipeline
      POST /api/v1/agents/trigger
      → Data Quality Agent runs Evidently → drift detected!
      → Model Eval Agent checks F1 → degradation detected!
      → Retraining Agent fine-tunes the model
      → Deployment Agent swaps to the new model
      → Grafana shows F1 recovering

4. Open Grafana (http://localhost:3000)
   → See the drift score spike
   → See F1 drop
   → See the retraining event
   → See F1 recover after deployment
```

### What the Interviewer Sees

A **live, self-healing ML system**:
- Model degrades → agents detect it → agents fix it → model recovers
- Zero human intervention
- Full observability via Grafana dashboards
- Experiment tracking via MLflow
- AI-queryable via MCP

---

## 20. Common Interview Questions

### Q: "Walk me through what happens when a prediction is made."

**A:** "The client sends a POST to /api/v1/predict with the ticket text. FastAPI validates the input using Pydantic. The text is tokenized using DistilBERT's tokenizer — it's converted to subword token IDs with padding and attention masks. The tokenized input goes through our DistilBERT model, which outputs logits for each of our 5 classes. We apply softmax to get probabilities, then take the argmax for the predicted class and the max probability as the confidence score. We update three Prometheus metrics: prediction count (by class), confidence histogram, and latency histogram. We also log the prediction to PostgreSQL for drift monitoring. The response includes the predicted label, confidence, and the full probability distribution across all 5 classes."

### Q: "How does your drift detection work?"

**A:** "We use Evidently AI to compare two data distributions: a reference dataset (our training data) and current production data (last 1000 predictions from PostgreSQL). Evidently runs statistical tests — Kolmogorov-Smirnov for numerical features and Jensen-Shannon divergence for categorical features. The output is a per-column drift score and an overall dataset drift flag. We track this as a Prometheus gauge so Grafana can show the drift score over time. When the Data Quality Agent detects drift, it passes this to the Model Evaluation Agent, which cross-references with performance metrics to decide if retraining is needed."

### Q: "Why did you use LangGraph instead of a simple if/else pipeline?"

**A:** "Three reasons. First, LangGraph gives us conditional routing — the pipeline can skip retraining if performance hasn't degraded, even if drift is detected. Second, each agent uses an LLM for nuanced reasoning. The Model Evaluation Agent doesn't just check thresholds; it considers drift and performance together and makes a judgment call. Third, LangGraph manages state cleanly — each agent reads from and writes to a typed state dict, making the data flow explicit and easy to debug."

### Q: "How do you prevent deploying a bad model?"

**A:** "Multiple safety layers. First, the Retraining Agent asks the LLM to evaluate whether the new model's metrics justify deployment. Second, the Deployment Agent only deploys if the new F1 is higher than the current (degraded) F1 AND the retraining summary contains 'DEPLOY'. Third, before swapping, we backup the current production model so we can rollback. Fourth, after deployment, the Performance Tracker resets its baseline, so we'll detect if the new model degrades quickly."

### Q: "What's MCP and why did you add it?"

**A:** "MCP is Model Context Protocol, an open standard from Anthropic for AI tool integration. It's like an API specifically designed for AI assistants to call. I added it because it lets any MCP-compatible AI (Claude, Cursor, etc.) query model status, trigger retraining, or classify tickets through natural language. Instead of opening Grafana, an engineer can ask Claude 'How's the model doing?' and get a live answer. It shows I understand the emerging AI tooling ecosystem, not just traditional ML infrastructure."

### Q: "How would you scale this for production?"

**A:** "Several changes. First, I'd use Kubernetes instead of Docker Compose for container orchestration. Second, I'd add a message queue (Kafka or RabbitMQ) between the prediction endpoint and the logging pipeline to handle traffic spikes. Third, I'd use a proper model serving framework like Triton or TorchServe for GPU-optimized inference. Fourth, I'd separate the agent pipeline into its own service that runs on a schedule (Airflow or a Kubernetes CronJob). Fifth, I'd add A/B testing with a traffic splitter to gradually roll out new models instead of hard swaps."

### Q: "What metrics would you alert on?"

**A:** "Three critical alerts. First, drift score exceeding 0.3 for more than 15 minutes — this means sustained data distribution shift. Second, F1 dropping more than 5% below baseline — direct model performance degradation. Third, prediction latency p99 exceeding 500ms — this impacts user experience. I'd also have warning-level alerts for confidence mean dropping and pipeline run failures."

---

## Quick Reference: File → Purpose Map

| File | One-Line Purpose |
|------|-----------------|
| `src/config.py` | All settings in one place (env vars, thresholds) |
| `src/ml/data_processor.py` | Text → tensors (tokenization, dataset splits) |
| `src/ml/model.py` | DistilBERT wrapper (load, predict, save) |
| `src/ml/train.py` | Training loop with MLflow logging |
| `src/ml/evaluate.py` | Compute classification metrics |
| `src/ml/predict.py` | End-to-end inference pipeline |
| `src/monitoring/drift_detector.py` | Evidently AI drift detection |
| `src/monitoring/performance_tracker.py` | Sliding window performance metrics |
| `src/monitoring/metrics_exporter.py` | Prometheus metric definitions |
| `src/agents/state.py` | Shared state schema for all agents |
| `src/agents/data_quality_agent.py` | Agent 1: drift check |
| `src/agents/model_eval_agent.py` | Agent 2: performance check |
| `src/agents/retraining_agent.py` | Agent 3: fine-tune model |
| `src/agents/deployment_agent.py` | Agent 4: model swap/rollback |
| `src/agents/orchestrator.py` | LangGraph pipeline wiring |
| `src/api/app.py` | FastAPI application setup |
| `src/api/routes/predict.py` | POST /predict endpoint |
| `src/api/routes/health.py` | GET /health endpoint |
| `src/api/routes/agents.py` | Agent status & trigger endpoints |
| `src/mcp/server.py` | MCP tools for AI clients |
| `src/storage/database.py` | PostgreSQL tables & queries |
| `src/storage/model_registry.py` | MLflow model versioning |
| `src/storage/s3_client.py` | Optional S3 artifact storage |

---

## Glossary

| Term | Definition |
|------|-----------|
| **Attention Mask** | Binary tensor telling the model which tokens are real vs padding |
| **Backpropagation** | Algorithm that computes gradients of loss w.r.t. parameters |
| **Batch** | Group of samples processed together (e.g., 16 tickets at once) |
| **Cross-Entropy Loss** | Loss function for classification — measures prediction error |
| **Drift** | When production data differs from training data |
| **Epoch** | One complete pass through all training data |
| **F1 Score** | Harmonic mean of precision and recall (our primary metric) |
| **Fine-Tuning** | Training a pre-trained model on a specific task |
| **Gradient** | Direction and magnitude of parameter updates |
| **Inference** | Using a trained model to make predictions |
| **Logits** | Raw model outputs before softmax |
| **MCP** | Model Context Protocol — standard for AI tool integration |
| **Overfitting** | Model memorizes training data but fails on new data |
| **Pre-training** | Training on massive general data (Wikipedia, books) |
| **Softmax** | Converts logits to probabilities summing to 1.0 |
| **Tensor** | Multi-dimensional array (the fundamental data structure in PyTorch) |
| **Tokenization** | Converting text to numerical token IDs |
| **Transformer** | Neural network architecture that uses self-attention |
| **Warmup** | Gradually increasing learning rate at training start |

---

*This document covers every major concept in the AgentOps project. Read it alongside the source code, and you'll be able to explain any component in detail during an interview.*
