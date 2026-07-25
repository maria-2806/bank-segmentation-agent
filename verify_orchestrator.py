import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_backend.agent_orchestrator import AgentOrchestrator

def main():
    print("Initializing AgentOrchestrator...")
    orchestrator = AgentOrchestrator(model_name="llama3.2")
    
    print("\nChecking Ollama status...")
    status = orchestrator.check_ollama_status()
    print(f"Ollama Status: {status}")
    
    if status.get('status') == 'offline':
        print("\n[WARNING] Ollama is currently OFFLINE.")
        print("Please ensure you have started Ollama (desktop app or service) and pulled the model.")
        print("Command to pull: ollama pull llama3.2")
        print("\nWe will skip live LLM tests, but basic orchestrator class instantiation succeeded.")
        sys.exit(0)
        
    print("\nOllama is ONLINE!")
    print(f"Available local models: {status.get('models')}")
    print(f"Configured model 'llama3.2' active: {status.get('configured_model_available')}")
    
    # Test query parsing
    test_query = "Segment customers into priority, regular and dormant based on balance and transaction frequency"
    print(f"\n--- Testing Query Parsing ---")
    print(f"Query: '{test_query}'")
    try:
        plan = orchestrator.parse_query(test_query)
        print(f"Parsed Plan JSON:\n{plan}")
        assert 'intent' in plan, "Parsed plan missing 'intent'"
        assert plan['intent'] == 'SEGMENTATION', "Parsed intent should be SEGMENTATION"
        print("SUCCESS: Query parsed correctly by local LLM!")
    except Exception as e:
        print(f"FAILED: Query parsing failed: {e}")
        sys.exit(1)
        
    print("\nSUCCESS: Stage 3 Orchestrator Verification Complete!")

if __name__ == "__main__":
    main()
