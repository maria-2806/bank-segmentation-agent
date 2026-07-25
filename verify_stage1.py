import pandas as pd
import os
import sys

def verify_stage1():
    print("==================================================")
    print("RUNNING STAGE 1 VERIFICATION")
    print("==================================================")
    
    csv_path = "banking_customers.csv"
    
    # 1. Check if dataset exists
    if not os.path.exists(csv_path):
        print(f"FAILED: {csv_path} not found.")
        sys.exit(1)
    print(f"SUCCESS: Found {csv_path}")
    
    # 2. Read dataset and check shape
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"FAILED: Could not read {csv_path}. Error: {e}")
        sys.exit(1)
        
    print(f"SUCCESS: Read dataset successfully. Shape: {df.shape}")
    
    # Assertions on dimensions
    expected_rows = 5000
    if len(df) != expected_rows:
        print(f"FAILED: Expected {expected_rows} rows, got {len(df)}")
        sys.exit(1)
    else:
        print(f"SUCCESS: Row count is exactly {expected_rows}")
        
    # 3. Check for required columns
    required_cols = [
        'customer_id', 'age', 'occupation', 'region', 'annual_income', 
        'account_balance', 'transaction_frequency', 'avg_transaction_amount', 
        'credit_score', 'debt_to_income', 'credit_card_limit', 
        'credit_card_utilization', 'online_login_frequency', 'tenure_months', 
        'has_personal_loan', 'has_credit_card', 'has_investment_account', 
        'investment_balance'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"FAILED: Missing required columns: {missing_cols}")
        sys.exit(1)
    else:
        print("SUCCESS: All required columns are present in the schema.")
        
    # 4. Check for null values
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        print(f"WARNING: Found {null_counts} null values in the dataset.")
    else:
        print("SUCCESS: No null values found in the dataset.")
        
    # 5. Check data bounds and type validation
    print("\nValidating data distributions and boundary logic:")
    
    # Credit score bounds
    min_cs = df['credit_score'].min()
    max_cs = df['credit_score'].max()
    print(f"  - Credit Score range: {min_cs} to {max_cs} (Expected: 300 to 850)")
    assert 300 <= min_cs and max_cs <= 850, "Credit Score out of logical bounds!"
    
    # Age bounds
    min_age = df['age'].min()
    max_age = df['age'].max()
    print(f"  - Customer Age range: {min_age} to {max_age} (Expected: 18 to 85)")
    assert 18 <= min_age and max_age <= 85, "Age out of logical bounds!"
    
    # Balance check
    min_bal = df['account_balance'].min()
    print(f"  - Minimum account balance: ${min_bal:,.2f} (Expected: >= 0.0)")
    assert min_bal >= 0, "Account balance cannot be negative!"
    
    # CC utilization check
    max_util = df['credit_card_utilization'].max()
    print(f"  - Maximum Credit Card utilization: {max_util*100:.1f}% (Expected: <= 105%)")
    assert max_util <= 1.05, "Credit Card utilization is too high!"
    
    print("\nSample Data Preview:")
    print(df.head(3).to_string(index=False))
    
    print("\n==================================================")
    print("STAGE 1 VERIFICATION PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    verify_stage1()
