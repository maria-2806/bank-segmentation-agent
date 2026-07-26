import pandas as pd
import numpy as np

# Canonical persona order: 0 Priority, 1 Regular, 2 Dormant, 3 Overleveraged, 4 Wealth/Investor
OCCUPATIONS = ['Professional', 'Blue Collar', 'Self-Employed', 'Retired', 'Student', 'Unemployed']
REGIONS = ['North', 'South', 'East', 'West', 'Central']
N_MONTHS = 3  # observation window that the transaction log spans


def _sample_persona(seg):
    """Draw one customer's raw (pre-bounds) attributes for a given persona."""
    if seg == 0:  # Priority
        f = dict(age=np.random.normal(45, 10), income=np.random.normal(140000, 25000),
                 balance=np.random.normal(85000, 15000), trans_freq=np.random.normal(25, 6),
                 avg_amt=np.random.normal(180, 40), credit=np.random.normal(760, 40),
                 dti=np.random.normal(0.2, 0.08), cc_limit=np.random.normal(25000, 5000),
                 cc_util=np.random.normal(0.25, 0.1), logins=np.random.normal(18, 4),
                 tenure=np.random.normal(84, 24),
                 has_loan=np.random.choice([0, 1], p=[0.7, 0.3]),
                 has_cc=np.random.choice([0, 1], p=[0.05, 0.95]),
                 has_inv=np.random.choice([0, 1], p=[0.4, 0.6]))
        f['inv_bal'] = np.random.normal(30000, 10000) if f['has_inv'] else 0.0
        f['occ'] = np.random.choice(OCCUPATIONS, p=[0.7, 0.05, 0.2, 0.05, 0.0, 0.0])
    elif seg == 1:  # Regular
        f = dict(age=np.random.normal(38, 12), income=np.random.normal(65000, 15000),
                 balance=np.random.normal(15000, 4000), trans_freq=np.random.normal(12, 4),
                 avg_amt=np.random.normal(55, 15), credit=np.random.normal(690, 45),
                 dti=np.random.normal(0.35, 0.1), cc_limit=np.random.normal(8000, 2000),
                 cc_util=np.random.normal(0.4, 0.15), logins=np.random.normal(10, 3),
                 tenure=np.random.normal(48, 18),
                 has_loan=np.random.choice([0, 1], p=[0.6, 0.4]),
                 has_cc=np.random.choice([0, 1], p=[0.2, 0.8]),
                 has_inv=np.random.choice([0, 1], p=[0.8, 0.2]))
        f['inv_bal'] = np.random.normal(5000, 2000) if f['has_inv'] else 0.0
        f['occ'] = np.random.choice(OCCUPATIONS, p=[0.4, 0.3, 0.15, 0.05, 0.08, 0.02])
    elif seg == 2:  # Dormant
        tf = np.random.choice([0, 1, 2], p=[0.6, 0.3, 0.1])
        f = dict(age=np.random.normal(48, 15), income=np.random.normal(50000, 12000),
                 balance=np.random.normal(650, 300), trans_freq=tf,
                 avg_amt=(np.random.normal(15, 8) if tf > 0 else 0.0),
                 credit=np.random.normal(610, 50), dti=np.random.normal(0.25, 0.12),
                 cc_limit=np.random.normal(3000, 1000), cc_util=np.random.normal(0.1, 0.08),
                 logins=np.random.normal(0.8, 0.8), tenure=np.random.normal(72, 30),
                 has_loan=np.random.choice([0, 1], p=[0.9, 0.1]),
                 has_cc=np.random.choice([0, 1], p=[0.5, 0.5]), has_inv=0)
        f['inv_bal'] = 0.0
        f['occ'] = np.random.choice(OCCUPATIONS, p=[0.2, 0.3, 0.1, 0.3, 0.05, 0.05])
    elif seg == 3:  # Overleveraged
        f = dict(age=np.random.normal(30, 8), income=np.random.normal(32000, 7000),
                 balance=np.random.normal(250, 180), trans_freq=np.random.normal(20, 5),
                 avg_amt=np.random.normal(14, 4), credit=np.random.normal(540, 45),
                 dti=np.random.normal(0.75, 0.15), cc_limit=np.random.normal(2500, 800),
                 cc_util=np.random.normal(0.88, 0.08), logins=np.random.normal(16, 5),
                 tenure=np.random.normal(24, 12),
                 has_loan=np.random.choice([0, 1], p=[0.15, 0.85]),
                 has_cc=np.random.choice([0, 1], p=[0.02, 0.98]), has_inv=0)
        f['inv_bal'] = 0.0
        f['occ'] = np.random.choice(OCCUPATIONS, p=[0.1, 0.4, 0.1, 0.0, 0.25, 0.15])
    else:  # Wealth/Investor
        f = dict(age=np.random.normal(54, 8), income=np.random.normal(220000, 40000),
                 balance=np.random.normal(260000, 50000), trans_freq=np.random.normal(4, 2),
                 avg_amt=np.random.normal(950, 300), credit=np.random.normal(790, 30),
                 dti=np.random.normal(0.12, 0.06), cc_limit=np.random.normal(45000, 10000),
                 cc_util=np.random.normal(0.08, 0.05), logins=np.random.normal(8, 3),
                 tenure=np.random.normal(110, 24),
                 has_loan=np.random.choice([0, 1], p=[0.9, 0.1]),
                 has_cc=np.random.choice([0, 1], p=[0.1, 0.9]), has_inv=1)
        f['inv_bal'] = np.random.normal(160000, 40000)
        f['occ'] = np.random.choice(OCCUPATIONS, p=[0.75, 0.0, 0.15, 0.1, 0.0, 0.0])
    return f


# Numeric attributes that get blended when a customer is "contaminated" toward another persona
_BLEND_KEYS = ['age', 'income', 'balance', 'trans_freq', 'avg_amt', 'credit', 'dti',
               'cc_limit', 'cc_util', 'logins', 'tenure', 'inv_bal']


def _blend(primary, other, alpha):
    """Interpolate the numeric attributes of two persona samples (alpha in [0,1])."""
    f = dict(primary)
    for k in _BLEND_KEYS:
        f[k] = (1 - alpha) * primary[k] + alpha * other[k]
    return f


def _simulate_transactions(cust_id, monthly_freq, avg_amount, start_day_offset):
    """
    Generate an individual transaction log for one customer across N_MONTHS.
    Monthly counts are Poisson-distributed around the customer's frequency, so the
    aggregated frequency is *derived*, not assumed. Returns a list of transaction rows.
    """
    txns = []
    lam = max(0.0, float(monthly_freq))
    for m in range(N_MONTHS):
        n = np.random.poisson(lam)
        for _ in range(n):
            amt = np.random.normal(avg_amount, max(1.0, avg_amount * 0.35))
            amt = round(max(0.01, amt), 2)
            day = start_day_offset + m * 30 + int(np.random.randint(0, 30))
            txns.append({
                'customer_id': cust_id,
                'day_index': day,
                'amount': amt,
                'txn_type': np.random.choice(['debit', 'credit'], p=[0.7, 0.3]),
            })
    return txns


def generate_banking_data(num_customers=5000, seed=42, contamination=0.18):
    """
    Generates a customer table plus a transaction log.

    - `contamination`: fraction of customers whose attributes are blended toward a
      second, random persona. This creates realistic boundary/ambiguous customers so
      the clustering task is no longer trivially separable.
    - transaction_frequency and avg_transaction_amount are DERIVED by aggregating the
      per-customer transaction log (group-by), demonstrating the ingestion pipeline the
      brief asks for, rather than being written directly.
    """
    np.random.seed(seed)

    segment_probs = [0.15, 0.45, 0.15, 0.15, 0.10]
    segments = np.random.choice([0, 1, 2, 3, 4], size=num_customers, p=segment_probs)

    rows = []
    all_txns = []

    for i in range(num_customers):
        cust_id = f"CUST{i+1:04d}"
        seg = int(segments[i])

        f = _sample_persona(seg)

        # Overlap: blend a fraction of customers toward a different persona
        if np.random.random() < contamination:
            other_seg = np.random.choice([s for s in range(5) if s != seg])
            alpha = np.random.uniform(0.30, 0.70)
            f = _blend(f, _sample_persona(other_seg), alpha)

        # Small global noise on every customer so nobody sits exactly on a centroid
        for k in ['balance', 'income', 'avg_amt']:
            f[k] *= np.random.normal(1.0, 0.06)

        # Bounds enforcement
        age = int(max(18, min(85, f['age'])))
        income = max(1000.0, float(f['income']))
        balance = max(0.0, float(f['balance']))
        target_freq = max(0, int(round(f['trans_freq'])))
        target_amt = max(0.0, float(f['avg_amt']))
        credit_score = int(max(300, min(850, f['credit'])))
        dti = max(0.0, min(2.5, float(f['dti'])))
        cc_limit = max(500.0, float(f['cc_limit']))
        cc_util = max(0.0, min(1.05, float(f['cc_util'])))
        tenure = int(max(1, f['tenure']))
        inv_bal = max(0.0, float(f['inv_bal']))

        # --- Transaction-level generation + aggregation (group-by) ---
        txns = _simulate_transactions(cust_id, target_freq, target_amt, start_day_offset=0)
        all_txns.extend(txns)
        if txns:
            tx_count = len(txns)
            derived_freq = int(round(tx_count / N_MONTHS))
            derived_avg = round(float(np.mean([t['amount'] for t in txns])), 2)
        else:
            derived_freq = 0
            derived_avg = 0.0

        online_logins = int(max(0, np.random.normal(f['logins'], 1.5)))

        rows.append({
            'customer_id': cust_id,
            'age': age,
            'occupation': f['occ'],
            'region': np.random.choice(REGIONS),
            'annual_income': round(income, 2),
            'account_balance': round(balance, 2),
            'transaction_frequency': derived_freq,        # aggregated from log
            'avg_transaction_amount': derived_avg,         # aggregated from log
            'credit_score': credit_score,
            'debt_to_income': round(dti, 3),
            'credit_card_limit': round(cc_limit, 2),
            'credit_card_utilization': round(cc_util, 3),
            'online_login_frequency': online_logins,
            'tenure_months': tenure,
            'has_personal_loan': int(f['has_loan']),
            'has_credit_card': int(f['has_cc']),
            'has_investment_account': int(f['has_inv']),
            'investment_balance': round(inv_bal, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv("banking_customers.csv", index=False)

    txn_df = pd.DataFrame(all_txns, columns=['customer_id', 'day_index', 'amount', 'txn_type'])
    txn_df.to_csv("transactions.csv", index=False)

    print(f"Generated {len(df)} customer records in banking_customers.csv")
    print(f"Generated {len(txn_df)} transactions in transactions.csv "
          f"(~{len(txn_df)//max(1,len(df))} per customer, {contamination:.0%} contamination)")
    print("\nMean features by planted persona:")
    print(df.groupby(segments)[['account_balance', 'transaction_frequency', 'annual_income']].mean().round(0))
    return df


if __name__ == "__main__":
    generate_banking_data()
