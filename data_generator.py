import pandas as pd
import numpy as np
import os

def generate_banking_data(num_customers=5000, seed=42):
    np.random.seed(seed)
    
    # Target segments representation in population
    # 0: Priority, 1: Regular, 2: Dormant, 3: Overleveraged, 4: Wealth/Investor
    segment_probs = [0.15, 0.45, 0.15, 0.15, 0.10]
    segments = np.random.choice([0, 1, 2, 3, 4], size=num_customers, p=segment_probs)
    
    data = []
    
    occupations = ['Professional', 'Blue Collar', 'Self-Employed', 'Retired', 'Student', 'Unemployed']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    for i in range(num_customers):
        cust_id = f"CUST{i+1:04d}"
        seg = segments[i]
        
        # Initialize default values
        age = 35
        income = 50000
        balance = 5000
        trans_freq = 10
        avg_trans_amount = 50
        credit_score = 650
        debt_to_income = 0.3
        cc_limit = 5000
        cc_utilization = 0.3
        online_logins = 8
        tenure = 36
        has_loan = 0
        has_cc = 1
        has_investment = 0
        investment_bal = 0
        
        # Segment-specific characteristics
        if seg == 0:  # Priority Customers
            age = int(np.random.normal(45, 10))
            income = float(np.random.normal(140000, 25000))
            balance = float(np.random.normal(85000, 15000))
            trans_freq = int(np.random.normal(25, 6))
            avg_trans_amount = float(np.random.normal(180, 40))
            credit_score = int(np.random.normal(760, 40))
            debt_to_income = float(np.random.normal(0.2, 0.08))
            cc_limit = float(np.random.normal(25000, 5000))
            cc_utilization = float(np.random.normal(0.25, 0.1))
            online_logins = int(np.random.normal(18, 4))
            tenure = int(np.random.normal(84, 24))
            has_loan = np.random.choice([0, 1], p=[0.7, 0.3])
            has_cc = np.random.choice([0, 1], p=[0.05, 0.95])
            has_investment = np.random.choice([0, 1], p=[0.4, 0.6])
            investment_bal = float(np.random.normal(30000, 10000)) if has_investment else 0.0
            occupation = np.random.choice(occupations, p=[0.7, 0.05, 0.2, 0.05, 0.0, 0.0])
            
        elif seg == 1:  # Regular Customers
            age = int(np.random.normal(38, 12))
            income = float(np.random.normal(65000, 15000))
            balance = float(np.random.normal(15000, 4000))
            trans_freq = int(np.random.normal(12, 4))
            avg_trans_amount = float(np.random.normal(55, 15))
            credit_score = int(np.random.normal(690, 45))
            debt_to_income = float(np.random.normal(0.35, 0.1))
            cc_limit = float(np.random.normal(8000, 2000))
            cc_utilization = float(np.random.normal(0.4, 0.15))
            online_logins = int(np.random.normal(10, 3))
            tenure = int(np.random.normal(48, 18))
            has_loan = np.random.choice([0, 1], p=[0.6, 0.4])
            has_cc = np.random.choice([0, 1], p=[0.2, 0.8])
            has_investment = np.random.choice([0, 1], p=[0.8, 0.2])
            investment_bal = float(np.random.normal(5000, 2000)) if has_investment else 0.0
            occupation = np.random.choice(occupations, p=[0.4, 0.3, 0.15, 0.05, 0.08, 0.02])
            
        elif seg == 2:  # Dormant Customers
            age = int(np.random.normal(48, 15))
            income = float(np.random.normal(50000, 12000))
            balance = float(np.random.normal(650, 300))
            trans_freq = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
            avg_trans_amount = float(np.random.normal(15, 8)) if trans_freq > 0 else 0.0
            credit_score = int(np.random.normal(610, 50))
            debt_to_income = float(np.random.normal(0.25, 0.12))
            cc_limit = float(np.random.normal(3000, 1000))
            cc_utilization = float(np.random.normal(0.1, 0.08))
            online_logins = int(np.random.normal(0.8, 0.8))
            online_logins = max(0, online_logins)
            tenure = int(np.random.normal(72, 30))
            has_loan = np.random.choice([0, 1], p=[0.9, 0.1])
            has_cc = np.random.choice([0, 1], p=[0.5, 0.5])
            has_investment = 0
            investment_bal = 0.0
            occupation = np.random.choice(occupations, p=[0.2, 0.3, 0.1, 0.3, 0.05, 0.05])
            
        elif seg == 3:  # Overleveraged Customers
            age = int(np.random.normal(30, 8))
            income = float(np.random.normal(32000, 7000))
            balance = float(np.random.normal(250, 180))
            trans_freq = int(np.random.normal(20, 5))
            avg_trans_amount = float(np.random.normal(14, 4))
            credit_score = int(np.random.normal(540, 45))
            debt_to_income = float(np.random.normal(0.75, 0.15))
            cc_limit = float(np.random.normal(2500, 800))
            cc_utilization = float(np.random.normal(0.88, 0.08))
            online_logins = int(np.random.normal(16, 5))
            tenure = int(np.random.normal(24, 12))
            has_loan = np.random.choice([0, 1], p=[0.15, 0.85])
            has_cc = np.random.choice([0, 1], p=[0.02, 0.98])
            has_investment = 0
            investment_bal = 0.0
            occupation = np.random.choice(occupations, p=[0.1, 0.4, 0.1, 0.0, 0.25, 0.15])
            
        else:  # Wealth/Investor Customers
            age = int(np.random.normal(54, 8))
            income = float(np.random.normal(220000, 40000))
            balance = float(np.random.normal(260000, 50000))
            trans_freq = int(np.random.normal(4, 2))
            avg_trans_amount = float(np.random.normal(950, 300))
            credit_score = int(np.random.normal(790, 30))
            debt_to_income = float(np.random.normal(0.12, 0.06))
            cc_limit = float(np.random.normal(45000, 10000))
            cc_utilization = float(np.random.normal(0.08, 0.05))
            online_logins = int(np.random.normal(8, 3))
            tenure = int(np.random.normal(110, 24))
            has_loan = np.random.choice([0, 1], p=[0.9, 0.1])
            has_cc = np.random.choice([0, 1], p=[0.1, 0.9])
            has_investment = 1
            investment_bal = float(np.random.normal(160000, 40000))
            occupation = np.random.choice(occupations, p=[0.75, 0.0, 0.15, 0.1, 0.0, 0.0])

        # Logical bounds enforcement
        age = max(18, min(85, age))
        income = max(1000.0, income)
        balance = max(0.0, balance)
        trans_freq = max(0, trans_freq)
        avg_trans_amount = max(0.0, avg_trans_amount)
        credit_score = max(300, min(850, credit_score))
        debt_to_income = max(0.0, min(2.5, debt_to_income))
        cc_limit = max(500.0, cc_limit)
        cc_utilization = max(0.0, min(1.05, cc_utilization))
        online_logins = max(0, online_logins)
        tenure = max(1, tenure)
        investment_bal = max(0.0, investment_bal)
        
        region = np.random.choice(regions)
        
        data.append({
            'customer_id': cust_id,
            'age': age,
            'occupation': occupation,
            'region': region,
            'annual_income': round(income, 2),
            'account_balance': round(balance, 2),
            'transaction_frequency': trans_freq,
            'avg_transaction_amount': round(avg_trans_amount, 2),
            'credit_score': credit_score,
            'debt_to_income': round(debt_to_income, 3),
            'credit_card_limit': round(cc_limit, 2),
            'credit_card_utilization': round(cc_utilization, 3),
            'online_login_frequency': online_logins,
            'tenure_months': tenure,
            'has_personal_loan': has_loan,
            'has_credit_card': has_cc,
            'has_investment_account': has_investment,
            'investment_balance': round(investment_bal, 2)
        })
        
    df = pd.DataFrame(data)
    # Save the file
    df.to_csv("banking_customers.csv", index=False)
    print(f"Generated {len(df)} customer records in banking_customers.csv")
    
    # Save small summary metrics to console for verification
    print("\nDataset Summary Stats:")
    print(df.groupby(segments)[['account_balance', 'transaction_frequency', 'annual_income']].mean())
    return df

if __name__ == "__main__":
    generate_banking_data()
