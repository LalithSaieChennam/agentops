"""Train the first model version.

Run this once to create the initial DistilBERT model for production.
"""

import shutil
from pathlib import Path

from src.ml.model import TicketClassifier
from src.ml.train import Trainer
from src.ml.data_processor import TicketDataProcessor


def main():
    print("=== AgentOps Initial Training ===\n")

    # Load and prepare data
    print("1. Loading and preparing data...")
    processor = TicketDataProcessor()
    train_dataset, val_dataset, test_dataset = processor.load_and_prepare()
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # Initialize model
    print("\n2. Initializing DistilBERT model...")
    model = TicketClassifier()

    # Train
    print("\n3. Training (this may take a few minutes)...")
    trainer = Trainer(
        model=model,
        learning_rate=2e-5,
        batch_size=16,
        num_epochs=3,
    )
    results = trainer.train(train_dataset, val_dataset, experiment_name="initial_training")

    print(f"\n4. Training complete!")
    print(f"   Best F1: {results['best_f1']:.4f}")
    print(f"   Model saved to: {results['model_path']}")

    # Copy best model to production
    production_path = Path("models/production")
    if production_path.exists():
        shutil.rmtree(production_path)
    shutil.copytree(results["model_path"], production_path)
    print(f"   Deployed to: models/production")

    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16)
    test_metrics = trainer._evaluate(test_loader)
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"   Per-class F1: {test_metrics['f1_per_class']}")

    print("\n=== Initial training complete! ===")
    print("You can now run: uvicorn src.api.app:app --reload")


if __name__ == "__main__":
    main()
