import subprocess
import sys
import os

def run_script(script_name):
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python" # Fallback
        
    print(f"\n==================================================")
    print(f"Running verification script: {script_name} using {python_exe}")
    print(f"==================================================")
    
    result = subprocess.run([python_exe, script_name], capture_output=False, text=True)
    if result.returncode != 0:
        print(f"Verification {script_name} FAILED with exit code {result.returncode}")
        return False
    print(f"Verification {script_name} PASSED.")
    return True

def main():
    print("Starting Customer Segmentation Agent Integration Verification...")
    
    # 1. Check data generator
    if not os.path.exists("banking_customers.csv"):
        print("Dataset not found. Generating customer dataset first...")
        python_exe = os.path.join(".venv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = "python"
        result = subprocess.run([python_exe, "data_generator.py"], capture_output=False)
        if result.returncode != 0:
            print("Failed to run data_generator.py")
            sys.exit(1)
            
    # 2. Run tools verification
    if not run_script("verify_tools.py"):
        sys.exit(1)
        
    # 3. Run orchestrator verification
    if not run_script("verify_orchestrator.py"):
        sys.exit(1)
        
    print("\nALL BACKEND MODULES SUCCESSFULLY VERIFIED!")
    print("Ready for FastAPI deployment and frontend connection.")

if __name__ == "__main__":
    main()
