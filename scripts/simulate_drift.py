"""Simulate data drift to trigger the agent pipeline.

This script gradually shifts the distribution of incoming tickets
to simulate a real-world scenario (e.g., a product launch causes
a surge in 'technical' tickets, or a pricing change causes
'billing' tickets to spike).
"""

import httpx
import random
import time
import asyncio

API_URL = "http://localhost:8000/api/v1/predict"

# Normal distribution of tickets
NORMAL_DISTRIBUTION = {
    "billing": 0.25,
    "technical": 0.25,
    "account": 0.20,
    "feature_request": 0.15,
    "general": 0.15,
}

# Drifted distribution (simulate product launch — technical tickets spike)
DRIFTED_DISTRIBUTION = {
    "billing": 0.10,
    "technical": 0.55,  # Spike!
    "account": 0.10,
    "feature_request": 0.20,
    "general": 0.05,
}

# Sample ticket templates per category
TICKET_TEMPLATES = {
    "billing": [
        "I was charged twice for my subscription this month",
        "Can I get a refund for the premium plan?",
        "My invoice shows the wrong amount",
        "How do I update my payment method?",
    ],
    "technical": [
        "The app crashes when I try to upload files",
        "API returns 500 error on the /users endpoint",
        "Integration with Slack stopped working after the update",
        "WebSocket connection drops every 30 seconds",
        "Getting CORS errors when calling from our frontend",
        "Memory usage spikes to 95% under load",
    ],
    "account": [
        "I can't reset my password",
        "How do I change my email address?",
        "I need to merge two accounts",
    ],
    "feature_request": [
        "Can you add dark mode to the dashboard?",
        "We need SSO support for our enterprise team",
        "Please add CSV export for reports",
    ],
    "general": [
        "What are your business hours?",
        "Is there a mobile app available?",
        "How do I contact sales?",
    ],
}


async def send_ticket(client: httpx.AsyncClient, category: str):
    """Send a simulated ticket to the prediction endpoint."""
    text = random.choice(TICKET_TEMPLATES[category])
    # Add some noise to make it more realistic
    text += f" [ref:{random.randint(1000, 9999)}]"

    response = await client.post(API_URL, json={"text": text})
    return response.json()


async def simulate(phase: str = "normal", count: int = 100, delay: float = 0.1):
    """Run a simulation phase.

    Args:
        phase: 'normal' or 'drift'
        count: Number of tickets to send
        delay: Seconds between tickets
    """
    distribution = NORMAL_DISTRIBUTION if phase == "normal" else DRIFTED_DISTRIBUTION
    categories = list(distribution.keys())
    weights = list(distribution.values())

    async with httpx.AsyncClient() as client:
        for i in range(count):
            category = random.choices(categories, weights=weights, k=1)[0]
            result = await send_ticket(client, category)

            if i % 25 == 0:
                print(f"[{phase}] Sent {i}/{count} tickets. "
                      f"Last prediction: {result['label']} ({result['confidence']:.2f})")

            await asyncio.sleep(delay)

    print(f"\n[{phase}] Phase complete. Sent {count} tickets.\n")


async def main():
    """Full demo: normal traffic -> drift -> watch agents react."""
    print("=== Phase 1: Normal Traffic (building baseline) ===")
    await simulate("normal", count=500, delay=0.05)

    print("Waiting 10 seconds for metrics to stabilize...\n")
    await asyncio.sleep(10)

    print("=== Phase 2: Drifted Traffic (simulating product launch) ===")
    await simulate("drift", count=500, delay=0.05)

    print("=== Phase 3: Triggering agent pipeline ===")
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8000/api/v1/agents/trigger")
        print(f"Pipeline triggered: {response.json()}")

    print("\nMonitor the pipeline at:")
    print("  Grafana:  http://localhost:3000")
    print("  MLflow:   http://localhost:5000")
    print("  API docs: http://localhost:8000/docs")


if __name__ == "__main__":
    asyncio.run(main())
