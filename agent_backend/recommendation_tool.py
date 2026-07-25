import pandas as pd
import numpy as np

def recommend_products_for_customer(customer_row, segment_id, segment_profiles):
    """
    Suggests financial products based on the customer's characteristics and segment profile.
    """
    recs = []
    
    balance = float(customer_row['account_balance'])
    credit_score = int(customer_row['credit_score'])
    dti = float(customer_row['debt_to_income'])
    cc_util = float(customer_row['credit_card_utilization'])
    has_loan = int(customer_row['has_personal_loan'])
    has_cc = int(customer_row['has_credit_card'])
    has_investment = int(customer_row['has_investment_account'])
    
    # Base recommendations on segment
    if segment_id == 0:  # Priority
        recs.append({
            'product': 'Premium Cashback Credit Card',
            'type': 'Up-Sell',
            'reason': 'High transaction frequency and excellent credit standing qualify you for our elite cashback rewards program.'
        })
        if not has_investment:
            recs.append({
                'product': 'Wealth Management Advisor Services',
                'type': 'Cross-Sell',
                'reason': 'Your substantial balance qualify you for a dedicated wealth planner to maximize your portfolio returns.'
            })
        else:
            recs.append({
                'product': 'High-Yield Investment Portfolio Review',
                'type': 'Cross-Sell',
                'reason': 'Maximize your active investments through our custom equity products.'
            })
            
    elif segment_id == 1:  # Regular
        if credit_score >= 680 and not has_cc:
            recs.append({
                'product': 'Gold Credit Card Account',
                'type': 'Cross-Sell',
                'reason': 'Great credit score qualifies you for low interest rates and 1.5% rewards points.'
            })
        if balance >= 10000 and not has_investment:
            recs.append({
                'product': 'Mutual Fund Starter Account',
                'type': 'Cross-Sell',
                'reason': 'Put your savings to work. Start investing with low minimum deposits and diversified mutual funds.'
            })
        if dti < 0.4 and not has_loan:
            recs.append({
                'product': 'Flexible Personal Line of Credit',
                'type': 'Cross-Sell',
                'reason': 'Your healthy debt-to-income ratio qualifies you for quick, low-interest funding options.'
            })
            
    elif segment_id == 2:  # Dormant
        recs.append({
            'product': 'Mobile App Login Campaign & Free Alerts',
            'type': 'Retention',
            'reason': 'Activate custom SMS notifications for real-time tracking and receive a $5 login reward.'
        })
        recs.append({
            'product': 'High-Yield Savings Promotion',
            'type': 'Retention',
            'reason': 'Earn 4.5% APY on new deposits of $500 or more to help reactivate your account balance.'
        })
        
    elif segment_id == 3:  # Overleveraged
        recs.append({
            'product': 'Debt Consolidation & Balance Transfer Program',
            'type': 'Retention',
            'reason': 'Combine high-interest credit card balances into a single low-interest monthly payment.'
        })
        recs.append({
            'product': 'Financial Wellness Counselor Session',
            'type': 'Support',
            'reason': 'Free 1-on-1 session to build a custom budget and establish credit building strategies.'
        })
        
    elif segment_id == 4:  # Wealth/Investor
        recs.append({
            'product': 'Private Banking Brokerage Account',
            'type': 'Up-Sell',
            'reason': 'Access premium stock-trading platforms and custom bond portfolio products.'
        })
        recs.append({
            'product': 'Tax-Advantaged Trust & Estate Services',
            'type': 'Cross-Sell',
            'reason': 'Preserve and transfer wealth with expert guidance on tax planning and legal structures.'
        })

    # Catch-all based on rules
    if credit_score < 580 and cc_util > 0.8:
        recs.append({
            'product': 'Secured Credit Builder Card',
            'type': 'Support',
            'reason': 'Improve your credit score safely by using a deposit-backed credit limit.'
        })
        
    return recs

def calculate_transition_steps(customer_row, target_segment_id, segment_profiles):
    """
    Calculates differences between a specific customer's features and the average values
    of a target segment (e.g. Priority), producing concrete behavioral transition targets.
    """
    if target_segment_id not in segment_profiles:
        return {"error": "Target segment profile does not exist."}
        
    target_profile = segment_profiles[target_segment_id]
    steps = []
    
    # Compare balance
    cust_bal = float(customer_row['account_balance'])
    target_bal = float(target_profile['account_balance'])
    if cust_bal < target_bal:
        diff_bal = target_bal - cust_bal
        steps.append({
            'metric': 'account_balance',
            'current': cust_bal,
            'target': target_bal,
            'difference': diff_bal,
            'action': f"Increase your account balance by ${diff_bal:,.2f} through consistent monthly deposits."
        })
        
    # Compare transaction frequency
    cust_freq = int(customer_row['transaction_frequency'])
    target_freq = int(target_profile['transaction_frequency'])
    if cust_freq < target_freq:
        diff_freq = target_freq - cust_freq
        steps.append({
            'metric': 'transaction_frequency',
            'current': cust_freq,
            'target': target_freq,
            'difference': diff_freq,
            'action': f"Increase monthly activity by making at least {diff_freq} more transactions (e.g., using card for daily coffee or bills)."
        })
        
    # Compare online login frequency
    cust_logins = int(customer_row['online_login_frequency'])
    target_logins = int(target_profile['online_login_frequency'])
    if cust_logins < target_logins:
        diff_logins = target_logins - cust_logins
        steps.append({
            'metric': 'online_login_frequency',
            'current': cust_logins,
            'target': target_logins,
            'difference': diff_logins,
            'action': f"Log in to online/mobile banking {diff_logins} more times per month to stay engaged with promotional rewards."
        })
        
    return steps
