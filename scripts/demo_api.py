"""Demo script showing how to use the API.

Start the API server first:
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Then run this script:
    uv run python scripts/demo_api.py
"""

import requests
import json

API_URL = "http://localhost:8000"


def demo_process_endpoint():
    """Demo the main /process endpoint."""
    print("=" * 60)
    print("DEMO: POST /process - Main Pipeline")
    print("=" * 60)

    # Example ticket
    ticket = {
        "subject": "Database connection timeout",
        "description": "Getting ERROR_TIMEOUT_429 when trying to sync large datasets. Connection times out after 30 seconds.",
        "product": "DataSync Pro",
        "priority": "high",
        "customer_tier": "enterprise",
    }

    print("\nTicket:")
    print(f"  Subject: {ticket['subject']}")
    print(f"  Product: {ticket['product']}")
    print(f"  Priority: {ticket['priority']}")

    # Call API
    response = requests.post(f"{API_URL}/process", json=ticket)

    if response.status_code == 200:
        result = response.json()
        print("\n✓ Response:")
        print(f"  Ticket ID: {result['ticket_id']}")
        print(f"  Predicted Category: {result['predicted_category']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Model Version: {result['model_version']}")
        print(f"\n  Similar Tickets Found: {len(result['similar_tickets'])}")

        if result["similar_tickets"]:
            print(f"\n  Top Solution:")
            top = result["similar_tickets"][0]
            print(f"    Ticket ID: {top['ticket_id']}")
            print(f"    Category: {top['category']}")
            print(f"    Similarity: {top['similarity_score']:.1%}")
            print(f"    Resolution: {top['resolution'][:100]}...")

        return result["ticket_id"]
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)
        return None


def demo_feedback_endpoint(ticket_id: str):
    """Demo the /feedback endpoint."""
    print("\n" + "=" * 60)
    print("DEMO: POST /feedback - Submit Agent Feedback")
    print("=" * 60)

    feedback = {
        "ticket_id": ticket_id,
        "true_category": "Technical Issue",
        "resolution_helpful": True,
        "comments": "The suggested solution worked perfectly!",
    }

    print("\nFeedback:")
    print(f"  Ticket ID: {feedback['ticket_id']}")
    print(f"  True Category: {feedback['true_category']}")
    print(f"  Resolution Helpful: {feedback['resolution_helpful']}")

    response = requests.post(f"{API_URL}/feedback", json=feedback)

    if response.status_code == 204:
        print("\n✓ Feedback submitted successfully!")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)


def demo_health_endpoint():
    """Demo the /health endpoint."""
    print("\n" + "=" * 60)
    print("DEMO: GET /health - System Health")
    print("=" * 60)

    response = requests.get(f"{API_URL}/health")

    if response.status_code == 200:
        health = response.json()
        print("\n✓ System Status:")
        print(f"  Status: {health['status']}")
        print(f"  Model Type: {health['model']['type']}")
        print(f"  Model Version: {health['model']['version']}")
        print(f"  Model Loaded: {health['model']['loaded']}")
        print(f"  RAG Documents: {health['rag']['documents']:,}")
        print(f"  Graph Nodes: {health['rag']['graph_nodes']:,}")
        print(f"  Predictions Tracked: {health['drift']['predictions_tracked']}")
    else:
        print(f"\n✗ Error: {response.status_code}")
        print(response.text)


def main():
    """Run all demos."""
    print("\n🚀 API Demo - Intelligent Support System\n")

    try:
        # 1. Check health
        demo_health_endpoint()

        # 2. Process a ticket
        ticket_id = demo_process_endpoint()

        # 3. Submit feedback
        if ticket_id:
            demo_feedback_endpoint(ticket_id)

        print("\n" + "=" * 60)
        print("✓ Demo complete!")
        print("=" * 60)
        print("\nTry the interactive docs at: http://localhost:8000/docs")
        print("View Prometheus metrics at: http://localhost:8000/metrics\n")

    except requests.exceptions.ConnectionError:
        print("\n✗ Error: Could not connect to API server.")
        print("Make sure the API is running:")
        print("  uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
