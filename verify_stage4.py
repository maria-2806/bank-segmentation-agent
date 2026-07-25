import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    if not os.path.exists("banking_customers.csv"):
        print("Dataset not found. Run data_generator.py first.")
        sys.exit(1)

    from fastapi.testclient import TestClient
    from main import app

    with TestClient(app) as client:
        print("Testing GET /api/status ...")
        r = client.get("/api/status")
        print(r.status_code, r.json())
        assert r.status_code == 200

        print("\nTesting GET /api/data ...")
        r = client.get("/api/data")
        print(r.status_code)
        assert r.status_code == 200
        assert "total_records" in r.json()

        print("\nTesting GET /api/customer/CUST0001 ...")
        r = client.get("/api/customer/CUST0001")
        print(r.status_code)
        assert r.status_code == 200

        print("\nTesting POST /api/chat ...")
        r = client.post("/api/chat", json={"query": "Give me an overview of the customer base"})
        print(r.status_code)
        if r.status_code != 200:
            print("NOTE: /api/chat needs Ollama running with the configured model:", r.text)
        else:
            assert "message" in r.json()

    print("\nSUCCESS: Stage 4 API Verification Complete!")


if __name__ == "__main__":
    main()
