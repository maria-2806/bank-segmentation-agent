import pandas as pd
import numpy as np
import json

def analyze_missing_values(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    return pd.DataFrame({
        'missing_count': missing,
        'missing_percentage': missing_pct
    }).to_dict(orient='index')

def compute_distributions(df, columns):
    dist_stats = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            stats = df[col].describe()
            dist_stats[col] = {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                '25%': float(stats['25%']),
                '50%': float(stats['50%']),
                '75%': float(stats['75%']),
                'max': float(stats['max'])
            }
    return dist_stats

def compute_correlation_matrix(df, numeric_columns):
    # Filter columns that are numeric and present in dataframe
    valid_cols = [col for col in numeric_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_cols:
        return {}
    
    corr_df = df[valid_cols].corr()
    
    # Structure for easy consumption in JS
    corr_matrix = {
        'columns': valid_cols,
        'matrix': corr_df.values.tolist()
    }
    return corr_matrix

def compute_histogram_bins(df, column, bins=20):
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        return []
    
    counts, edges = np.histogram(df[column].dropna(), bins=bins)
    
    return [
        {
            'bin_start': float(edges[i]),
            'bin_end': float(edges[i+1]),
            'count': int(counts[i])
        }
        for i in range(len(counts))
    ]

def get_categorical_distributions(df, columns):
    distributions = {}
    for col in columns:
        if col in df.columns:
            val_counts = df[col].value_counts()
            distributions[col] = val_counts.to_dict()
    return distributions

def run_full_eda(df_path="banking_customers.csv"):
    try:
        df = pd.read_csv(df_path)
    except FileNotFoundError:
        return {"error": f"File {df_path} not found. Please generate the data first."}
        
    numeric_cols = [
        'age', 'annual_income', 'account_balance', 'transaction_frequency', 
        'avg_transaction_amount', 'credit_score', 'debt_to_income', 
        'credit_card_limit', 'credit_card_utilization', 'online_login_frequency', 
        'tenure_months', 'investment_balance'
    ]
    
    categorical_cols = ['occupation', 'region']
    
    missing_analysis = analyze_missing_values(df)
    distributions = compute_distributions(df, numeric_cols)
    correlation = compute_correlation_matrix(df, numeric_cols)
    cat_distributions = get_categorical_distributions(df, categorical_cols)
    
    # Custom charts binning data for frontend
    balance_bins = compute_histogram_bins(df, 'account_balance', bins=30)
    transactions_bins = compute_histogram_bins(df, 'transaction_frequency', bins=20)
    income_bins = compute_histogram_bins(df, 'annual_income', bins=20)
    
    # Scatter data for balance vs transaction frequency (sample to 1000 points to prevent frontend lag)
    sample_df = df.sample(min(1000, len(df)), random_state=42)
    scatter_data = sample_df[['customer_id', 'account_balance', 'transaction_frequency', 'annual_income']].to_dict(orient='records')
    
    return {
        'total_records': len(df),
        'missing_values': missing_analysis,
        'summary_statistics': distributions,
        'correlation_matrix': correlation,
        'categorical_distributions': cat_distributions,
        'chart_data': {
            'balance_histogram': balance_bins,
            'transaction_histogram': transactions_bins,
            'income_histogram': income_bins,
            'scatter_balance_vs_transactions': scatter_data
        }
    }
