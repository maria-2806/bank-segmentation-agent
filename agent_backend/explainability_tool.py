import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree

def fit_explainability_tree(X, labels, feature_names, max_depth=3):
    """
    Fits a shallow decision tree to classify the cluster labels,
    allowing us to extract clean, global rules for the segment boundaries.
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X, labels)
    
    # Feature importances
    importances = dt.feature_importances_
    feat_imp = [
        {'feature': name, 'importance': float(imp)}
        for name, imp in zip(feature_names, importances)
        if imp > 0
    ]
    feat_imp = sorted(feat_imp, key=lambda x: x['importance'], reverse=True)
    
    return dt, feat_imp

def extract_decision_rules(dt, feature_names, class_names=None):
    """
    Traverses the Decision Tree to return lists of rules for each leaf/segment.
    """
    tree_ = dt.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    
    rules = {}
    
    def recurse(node, depth, path_rules):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left child (Feature <= threshold)
            left_rule = f"{name} <= {threshold:.2f}"
            recurse(tree_.children_left[node], depth + 1, path_rules + [left_rule])
            
            # Right child (Feature > threshold)
            right_rule = f"{name} > {threshold:.2f}"
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
    their cluster assignment.
    """
    # Transform customer features
    X_cust = pipeline.transform(df_customer)
    
    node = 0
    tree_ = dt_model.tree_
    path_taken = []
    
    # Follow decision path
    while tree_.feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = tree_.feature[node]
        feat_name = feature_names[feat_idx]
        threshold = tree_.threshold[node]
        
        # Get raw value for explaining (original scale if possible, otherwise scaled value)
        # Note: raw value should be obtained from df_customer
        val_in_df = df_customer.iloc[0].get(feat_name, None)
        # If feat_name is engineered (e.g. balance_to_income_ratio), calculate it
        if val_in_df is None:
            if feat_name == 'balance_to_income_ratio':
                val_in_df = df_customer.iloc[0]['account_balance'] / (df_customer.iloc[0]['annual_income'] + 1)
            elif feat_name == 'investment_ratio':
                val_in_df = df_customer.iloc[0]['investment_balance'] / (df_customer.iloc[0]['account_balance'] + df_customer.iloc[0]['investment_balance'] + 1)
            else:
                val_in_df = X_cust[0][feat_idx] # Fallback to scaled value
                
        scaled_val = X_cust[0][feat_idx]
        
        if scaled_val <= threshold:
            path_taken.append({
                'feature': feat_name,
                'raw_value': float(val_in_df) if isinstance(val_in_df, (int, float, np.integer, np.floating)) else val_in_df,
                'comparison': '<=',
                'threshold': float(threshold),
                'decision': f"{feat_name} ({val_in_df:.2f} if numeric else value) is less than or equal to threshold"
            })
            node = tree_.children_left[node]
        else:
            path_taken.append({
                'feature': feat_name,
                'raw_value': float(val_in_df) if isinstance(val_in_df, (int, float, np.integer, np.floating)) else val_in_df,
                'comparison': '>',
                'threshold': float(threshold),
                'decision': f"{feat_name} ({val_in_df:.2f} if numeric else value) is greater than threshold"
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
