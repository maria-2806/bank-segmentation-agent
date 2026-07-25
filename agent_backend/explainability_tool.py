import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree

def fit_explainability_tree(X, labels, feature_names, max_depth=3):
    """
    Fits a shallow decision tree to approximate the K-Means cluster assignments,
    yielding human-interpretable classification boundaries.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X, labels)
    
    # Feature importances
    importances = dict(zip(feature_names, dt.feature_importances_))
    return dt, importances

def unscale_threshold(feat_name, feat_idx, threshold, scaler=None):
    """
    Converts a Z-score scaled threshold back to its raw feature scale using standard scaler mean/scale.
    """
    if scaler is not None and hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        if feat_idx < len(scaler.mean_):
            mean = scaler.mean_[feat_idx]
            scale = scaler.scale_[feat_idx]
            return float(threshold * scale + mean)
    return float(threshold)

def format_rule_threshold(feat_name, raw_val):
    """
    Formats a raw threshold value dynamically (currency $, integer counts, or percentages).
    """
    feat_lower = feat_name.lower()
    if any(k in feat_lower for k in ['balance', 'income', 'limit', 'amount']):
        return f"${raw_val:,.0f}"
    elif any(k in feat_lower for k in ['frequency', 'age', 'score', 'tenure', 'logins', 'count']):
        return f"{raw_val:.1f}"
    elif 'ratio' in feat_lower or 'utilization' in feat_lower:
        return f"{raw_val:.2f}"
    else:
        return f"{raw_val:.2f}"

def extract_decision_rules(dt, feature_names, class_names=None, scaler=None):
    """
    Traverses the Decision Tree to return lists of human-readable rules for each leaf/segment,
    un-scaling thresholds back to raw feature units ($ dollars, transaction counts).
    """
    tree_ = dt.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = {}
    
    def recurse(node, depth, path_rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat_idx = tree_.feature[node]
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            raw_thresh = unscale_threshold(name, feat_idx, threshold, scaler)
            thresh_str = format_rule_threshold(name, raw_thresh)
            
            # Left child (Feature <= threshold)
            left_rule = f"{name} <= {thresh_str}"
            recurse(tree_.children_left[node], depth + 1, path_rules + [left_rule])
            
            # Right child (Feature > threshold)
            right_rule = f"{name} > {thresh_str}"
            recurse(tree_.children_right[node], depth + 1, path_rules + [right_rule])
        else:
            # Leaf node
            value = tree_.value[node][0]
            assigned_class = int(np.argmax(value))
            
            if assigned_class not in rules:
                rules[assigned_class] = []
            
            rules[assigned_class].append({
                'rule': " AND ".join(path_rules),
                'samples': int(tree_.n_node_samples[node]),
                'class_distribution': [int(val) for val in value]
            })
            
    recurse(0, 1, [])
    return rules

def explain_single_customer(df_customer, pipeline, dt_model, feature_names):
    """
    Traces a single customer's path through the Decision Tree model to explain
    their cluster assignment with unscaled threshold values.
    """
    scaler = pipeline.named_steps['scaler'] if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps else None
    X_cust = pipeline.transform(df_customer)
    
    node = 0
    tree_ = dt_model.tree_
    path_taken = []
    
    # Follow decision path
    while tree_.feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = tree_.feature[node]
        feat_name = feature_names[feat_idx]
        threshold = tree_.threshold[node]
        
        # Unscale threshold to original units ($ dollars / counts)
        raw_thresh = unscale_threshold(feat_name, feat_idx, threshold, scaler)
        thresh_str = format_rule_threshold(feat_name, raw_thresh)
        
        # Get raw value for explaining from df_customer
        val_in_df = df_customer.iloc[0].get(feat_name, None)
        if val_in_df is None:
            if feat_name == 'balance_to_income_ratio':
                val_in_df = df_customer.iloc[0]['account_balance'] / (df_customer.iloc[0]['annual_income'] + 1)
            elif feat_name == 'investment_ratio':
                val_in_df = df_customer.iloc[0]['investment_balance'] / (df_customer.iloc[0]['account_balance'] + df_customer.iloc[0]['investment_balance'] + 1)
            else:
                val_in_df = X_cust[0][feat_idx] # Fallback
                
        scaled_val = X_cust[0][feat_idx]
        raw_val_float = float(val_in_df) if isinstance(val_in_df, (int, float, np.integer, np.floating)) else val_in_df
        val_str = format_rule_threshold(feat_name, raw_val_float) if isinstance(raw_val_float, (int, float)) else str(val_str)
        
        if scaled_val <= threshold:
            path_taken.append({
                'feature': feat_name,
                'raw_value': raw_val_float,
                'comparison': '<=',
                'threshold': raw_thresh,
                'decision': f"{feat_name} ({val_str}) <= threshold boundary {thresh_str}"
            })
            node = tree_.children_left[node]
        else:
            path_taken.append({
                'feature': feat_name,
                'raw_value': raw_val_float,
                'comparison': '>',
                'threshold': raw_thresh,
                'decision': f"{feat_name} ({val_str}) > threshold boundary {thresh_str}"
            })
            node = tree_.children_right[node]
            
    leaf_distribution = [int(v) for v in tree_.value[node][0]]
    assigned_class = int(np.argmax(tree_.value[node][0]))
    
    return {
        'assigned_segment': assigned_class,
        'path': path_taken,
        'leaf_distribution': leaf_distribution
    }

def get_cluster_profiles(df, segment_col='segment_id'):
    """
    Computes mean values of key metrics for each segment to build profile cards.
    """
    profile_cols = [
        'age', 'annual_income', 'account_balance', 'transaction_frequency',
        'avg_transaction_amount', 'credit_score', 'debt_to_income',
        'credit_card_utilization', 'online_login_frequency', 'tenure_months',
        'has_personal_loan', 'has_credit_card', 'has_investment_account', 'investment_balance'
    ]
    
    # Filter valid columns present in df
    cols = [col for col in profile_cols if col in df.columns]
    
    # Group and average
    grouped = df.groupby(segment_col)[cols].mean()
    
    # Format output as dictionary
    profiles = {}
    for idx, row in grouped.iterrows():
        profiles[int(idx)] = row.to_dict()
        
    return profiles
