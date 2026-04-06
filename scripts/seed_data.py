"""Generate initial training data for the ticket classifier.

Downloads a public dataset and maps it to our support ticket
label schema, then saves it as CSV for training.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import random


def generate_synthetic_tickets(count: int = 2000) -> pd.DataFrame:
    """Generate synthetic support tickets for each category."""
    templates = {
        "billing": [
            "I was charged twice for my subscription this month",
            "Can I get a refund for the premium plan?",
            "My invoice shows the wrong amount",
            "How do I update my payment method?",
            "I need to cancel my subscription and get a prorated refund",
            "There's an unauthorized charge on my account",
            "Can you explain the charges on my latest bill?",
            "I want to switch from monthly to annual billing",
            "My coupon code isn't working at checkout",
            "When will I receive my refund?",
        ],
        "technical": [
            "The app crashes when I try to upload files",
            "API returns 500 error on the /users endpoint",
            "Integration with Slack stopped working after the update",
            "WebSocket connection drops every 30 seconds",
            "Getting CORS errors when calling from our frontend",
            "Memory usage spikes to 95% under load",
            "The dashboard takes 30 seconds to load",
            "Push notifications stopped working on iOS",
            "Database queries are timing out",
            "SSL certificate error when connecting to the API",
        ],
        "account": [
            "I can't reset my password",
            "How do I change my email address?",
            "I need to merge two accounts",
            "My account was locked after too many login attempts",
            "How do I enable two-factor authentication?",
            "I want to delete my account and all my data",
            "Can I transfer my account to a colleague?",
            "My profile picture won't upload",
            "I need to update my company name on the account",
            "How do I add team members to my organization?",
        ],
        "feature_request": [
            "Can you add dark mode to the dashboard?",
            "We need SSO support for our enterprise team",
            "Please add CSV export for reports",
            "Can you integrate with Jira for project tracking?",
            "We'd love to have webhooks for real-time notifications",
            "Please add support for custom fields in forms",
            "Can you add a mobile app for Android?",
            "We need an API for bulk data import",
            "Please add role-based access control",
            "Can you support multiple languages?",
        ],
        "general": [
            "What are your business hours?",
            "Is there a mobile app available?",
            "How do I contact sales?",
            "Where can I find the documentation?",
            "What's your uptime SLA?",
            "Do you offer a free trial?",
            "What payment methods do you accept?",
            "How does your pricing work?",
            "Can you tell me more about your enterprise plan?",
            "Where is your data center located?",
        ],
    }

    rows = []
    categories = list(templates.keys())

    for _ in range(count):
        category = random.choice(categories)
        text = random.choice(templates[category])
        # Add some variation
        text += f" [ticket-{random.randint(1000, 99999)}]"
        rows.append({"ticket_text": text, "label_name": category})

    df = pd.DataFrame(rows)
    # Map to label IDs
    label_map = {"billing": 0, "technical": 1, "account": 2, "feature_request": 3, "general": 4}
    df["label"] = df["label_name"].map(label_map)
    return df


def download_public_dataset() -> pd.DataFrame:
    """Download and remap a public dataset for initial training."""
    print("Downloading ag_news dataset...")
    dataset = load_dataset("ag_news", split="train[:10000]")
    df = pd.DataFrame(dataset)

    # Map ag_news labels (0-3) to our labels
    ag_to_ticket = {0: "general", 1: "technical", 2: "billing", 3: "feature_request"}
    df["label_name"] = df["label"].map(ag_to_ticket)

    label_map = {"billing": 0, "technical": 1, "account": 2, "feature_request": 3, "general": 4}
    df["label"] = df["label_name"].map(label_map)
    df = df.rename(columns={"text": "ticket_text"})

    return df


def main():
    """Generate and save training data."""
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("Generating synthetic tickets...")
    synthetic = generate_synthetic_tickets(2000)
    synthetic.to_csv(output_dir / "raw" / "synthetic_tickets.csv", index=False)
    print(f"  Saved {len(synthetic)} synthetic tickets")

    # Download public dataset
    public = download_public_dataset()
    public.to_csv(output_dir / "raw" / "public_dataset.csv", index=False)
    print(f"  Saved {len(public)} public dataset entries")

    # Combine for training
    combined = pd.concat([synthetic, public], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined.to_csv(output_dir / "processed" / "training_data.csv", index=False)
    print(f"  Saved {len(combined)} combined training samples")

    # Save reference data for drift detection
    combined.to_csv(output_dir / "reference" / "reference_data.csv", index=False)
    print(f"  Saved reference data for drift detection")

    print("\nData generation complete!")
    print(f"  Label distribution:\n{combined['label_name'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
