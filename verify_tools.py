import pandas as pd
import numpy as np
import os
import sys

# Add backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from agent_backend.eda_tool import run_full_eda
    from agent_backend.feature_engineering_tool import get_preprocessed_features
    from agent_backend.segmentation_tool import run_kmeans_clustering
    from agent_backend.explainability_tool import fit_explainability_tree, extract_decision_rules, explain_single_customer, get_cluster_profiles
    from agent_backend.recommendation_tool import recommend_products_for_customer, calculate_transition_steps
    print("SUCCESS: Imported all analytic tools.")
except Exception as e:
    print(f"FAILED: Import error: {e}")
    sys.exit(1)

def run_verification():
    csv_path = "banking_customers.csv"
    if not os.path.exists(csv_path):
        print(f"FAILED: {csv_path} not found. Please run data_generator.py first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns.")
    
    # 1. Verify EDA
    print("\n--- 1. Verifying EDA Tool ---")
    eda_results = run_full_eda(csv_path)
    assert 'total_records' in eda_results, "EDA results missing 'total_records'"
    assert 'chart_data' in eda_results, "EDA results missing 'chart_data'"
    print(f"EDA verified. Total records analyzed: {eda_results['total_records']}")
    print(f"Scatter plot data samples: {len(eda_results['chart_data']['scatter_balance_vs_transactions'])}")
    
    # 2. Verify Feature Engineering
    print("\n--- 2. Verifying Feature Engineering Tool ---")
    features = ['account_balance', 'transaction_frequency', 'annual_income', 'credit_score', 'occupation']
    X, feat_names, pipeline, df_engineered = get_preprocessed_features(df, features)
    assert X.shape[0] == len(df), "Engineered feature rows mismatch"
    assert len(feat_names) > 0, "Feature names empty"
    print(f"Feature engineering verified. X shape: {X.shape}, Columns: {feat_names}")
    
    # 3. Verify Segmentation
    print("\n--- 3. Verifying Segmentation Tool ---")
    seg_results = run_kmeans_clustering(df_engineered, X, feat_names, num_clusters=5)
    df_segmented = seg_results['df']
    assert 'segment_id' in df_segmented.columns, "Segment column missing in df"
    assert 'is_outlier' in df_segmented.columns, "Outlier column missing in df"
    assert 'is_boundary' in df_segmented.columns, "Boundary column missing in df"
    print(f"Clustering verified. Silhouette Score: {seg_results['silhouette_score']:.4f}")
    print(f"Segment sizes: {seg_results['cluster_sizes']}")
    
    # 4. Verify Explainability
    print("\n--- 4. Verifying Explainability Tool ---")
    dt_model, importances = fit_explainability_tree(X, seg_results['labels'], feat_names)
    rules = extract_decision_rules(dt_model, feat_names)
    assert len(rules) > 0, "No rules extracted"
    
    profiles = get_cluster_profiles(df_segmented)
    assert len(profiles) == 5, "Profiles counts mismatch"
    
    # Explain single customer
    single_cust = df_segmented.iloc[[0]]
    cust_explanation = explain_single_customer(single_cust, pipeline, dt_model, feat_names)
    assert cust_explanation['assigned_segment'] == single_cust['segment_id'].values[0], "Assigned segment in trace does not match"
    print("Explainability verified. Top features driving classification:")
    for imp in importances[:3]:
        print(f"  {imp['feature']}: {imp['importance']:.4f}")
    print(f"Global decision tree rules extracted for {len(rules)} segments.")
    
    # 5. Verify Recommendations
    print("\n--- 5. Verifying Recommendation Tool ---")
    cust_row = df_segmented.iloc[0]
    segment = int(cust_row['segment_id'])
    recs = recommend_products_for_customer(cust_row, segment, profiles)
    assert len(recs) > 0, "No recommendations generated"
    print(f"Recommendations verified for customer {cust_row['customer_id']} (Segment {segment}):")
    for r in recs:
        print(f"  [{r['type']}] {r['product']} - Reason: {r['reason'][:60]}...")
        
    # Verify transition
    if segment != 0: # If not already Priority
        steps = calculate_transition_steps(cust_row, 0, profiles)
        print(f"Transition steps to Segment 0 (Priority):")
        for s in steps:
            print(f"  - Need to adjust {s['metric']}: current {s['current']:.1f} -> target {s['target']:.1f} ({s['action']})")
            
    print("\nSUCCESS: All analytical tools verified successfully!")

if __name__ == "__main__":
    run_verification()
