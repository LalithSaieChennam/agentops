"""Run the full AgentOps demo end-to-end.

This script:
1. Seeds training data
2. Trains the initial model
3. Starts normal traffic
4. Introduces drift
5. Triggers the agent pipeline
6. Shows the results
"""

import asyncio
import httpx
import time
import sys

API_URL = "http://localhost:8000"


async def check_health():
    """Check if the API is running."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/api/v1/health")
            return response.status_code == 200
        except Exception:
            return False


async def send_predictions(count: int, phase: str = "normal"):
    """Send simulated predictions."""
    from scripts.simulate_drift import simulate
    await simulate(phase, count=count, delay=0.02)


async def main():
    print("=" * 60)
    print("  AgentOps — Full Demo")
    print("=" * 60)

    # Check API
    print("\n[1/5] Checking API health...")
    if not await check_health():
        print("  ERROR: API is not running!")
        print("  Start with: docker-compose up")
        print("  Or: uvicorn src.api.app:app --reload")
        sys.exit(1)
    print("  API is healthy!")

    # Normal traffic
    print("\n[2/5] Sending normal traffic (building baseline)...")
    await send_predictions(300, "normal")

    # Check status
    print("\n[3/5] Checking model status...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/v1/agents/status")
        print(f"  Status: {response.json()}")

    # Introduce drift
    print("\n[4/5] Introducing data drift (simulating product launch)...")
    await send_predictions(300, "drift")

    # Trigger pipeline
    print("\n[5/5] Triggering agent pipeline...")
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_URL}/api/v1/agents/trigger")
        print(f"  Pipeline triggered: {response.json()}")

    # Wait for pipeline to complete
    print("\n  Waiting for pipeline to complete...")
    await asyncio.sleep(30)

    # Check final status
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_URL}/api/v1/agents/status")
        status = response.json()
        print(f"\n  Final Status:")
        print(f"    Drift Detected: {status.get('drift_detected')}")
        print(f"    Current F1: {status.get('current_f1')}")
        print(f"    Last Action: {status.get('last_action')}")
        print(f"    Summary: {status.get('summary')}")

    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    print("\n  View dashboards at:")
    print("    Grafana:  http://localhost:3000  (admin/admin)")
    print("    MLflow:   http://localhost:5000")
    print("    API docs: http://localhost:8000/docs")
    print("    Prometheus: http://localhost:9090")


if __name__ == "__main__":
    asyncio.run(main())
