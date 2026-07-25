import os
import httpx
import json
import pandas as pd
import numpy as np
import traceback

from agent_backend.eda_tool import run_full_eda
from agent_backend.feature_engineering_tool import get_preprocessed_features
from agent_backend.segmentation_tool import run_kmeans_clustering
from agent_backend.explainability_tool import fit_explainability_tree, extract_decision_rules, explain_single_customer, get_cluster_profiles
from agent_backend.recommendation_tool import recommend_products_for_customer, calculate_transition_steps

class AgentOrchestrator:
    def __init__(self, ollama_host="http://localhost:11434", model_name="llama3.2"):
        self.ollama_host = ollama_host.rstrip('/')
        self.model_name = model_name
        self.client = httpx.Client(timeout=60.0) # Local LLM can be slow
        
    def check_ollama_status(self):
        try:
            resp = self.client.get(f"{self.ollama_host}/api/tags")
            if resp.status_code == 200:
                models = [m['name'] for m in resp.json().get('models', [])]
                return {
                    'status': 'online',
                    'models': models,
                    'configured_model_available': any(self.model_name in m for m in models)
                }
        except Exception as e:
            return {'status': 'offline', 'error': str(e)}
            
    def _call_ollama(self, prompt, format_json=False):
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        if format_json:
            payload["format"] = "json"
            
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        
        response_text = resp.json().get("response", "")
        return response_text
        
    def parse_query(self, query):
        prompt = f"""You are the Query Planner for a Banking Customer Analytics Agent.
Parse the user's natural language query and output a JSON object describing the intent and variables to extract.

Supported Intents:
1. "EDA": General overview, statistics, distributions, correlations of the dataset.
2. "SEGMENTATION": Grouping, dividing, or clustering customers based on columns.
3. "EXPLAIN": Explaining global segment rules, distinguishing features, or why a single customer is in their group.
4. "RECOMMEND": Personalized banking products, marketing campaigns, or segment transition advice.

Recognized Numeric Columns:
- 'age', 'annual_income', 'account_balance', 'transaction_frequency', 'avg_transaction_amount', 'credit_score', 'debt_to_income', 'credit_card_limit', 'credit_card_utilization', 'online_login_frequency', 'tenure_months', 'investment_balance'

Output JSON Schema:
{{
  "intent": "EDA" | "SEGMENTATION" | "EXPLAIN" | "RECOMMEND",
  "columns": ["list of columns relevant to the query"],
  "num_clusters": null or integer (if segmentation or grouping is requested, default to 3 if not specified),
  "customer_id": null or string (e.g. "CUST0104" if a specific customer is mentioned),
  "question_type": "global" | "single_customer" | "aggregation"
}}

Rules:
- If the user asks about priority/dormant segment criteria globally, intent is "EXPLAIN" and question_type is "global".
- If the user asks about a specific customer (e.g. "CUST0005"), intent is "EXPLAIN" or "RECOMMEND" and question_type is "single_customer", customer_id is "CUST0005".
- If they ask "What is the average transaction size for regular customers?", intent is "EDA", question_type is "aggregation", columns is ["avg_transaction_amount", "transaction_frequency"].

Query: "{query}"
JSON:"""

        response_text = self._call_ollama(prompt, format_json=True)
        try:
            return json.loads(response_text)
        except Exception:
            # Basic fallback parser if JSON fails
            q_lower = query.lower()
            plan = {"intent": "EDA", "columns": [], "num_clusters": 3, "customer_id": None, "question_type": "global"}
            if "segment" in q_lower or "cluster" in q_lower or "group" in q_lower:
                plan["intent"] = "SEGMENTATION"
            elif "why" in q_lower or "explain" in q_lower or "basis" in q_lower or "rule" in q_lower:
                plan["intent"] = "EXPLAIN"
            elif "recommend" in q_lower or "convert" in q_lower or "offer" in q_lower:
                plan["intent"] = "RECOMMEND"
                
            # Extract customer ID
            import re
            match = re.search(r'cust\d+', q_lower)
            if match:
                plan["customer_id"] = match.group(0).upper()
                plan["question_type"] = "single_customer"
            return plan

    def execute_query(self, query, csv_path="banking_customers.csv"):
        thoughts = []
        thoughts.append("Consulting Ollama LLM to parse query intent...")
        
        try:
            plan = self.parse_query(query)
        except Exception as e:
            raise RuntimeError(f"Ollama Connection Failed. Please ensure Ollama is running and has model {self.model_name}. Details: {e}")
            
        intent = plan.get("intent", "EDA")
        cols = plan.get("columns", [])
        num_clusters = plan.get("num_clusters", 3)
        cust_id = plan.get("customer_id")
        q_type = plan.get("question_type", "global")
        
        # Ensure we have column defaults
        if not cols:
            cols = ['account_balance', 'transaction_frequency', 'annual_income', 'credit_score', 'tenure_months']
            
        thoughts.append(f"Parsed intent: {intent} (Targets: {', '.join(cols) if cols else 'all'}, Clusters: {num_clusters}, CustID: {cust_id})")
        
        # Load dataset
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Customer dataset {csv_path} not found. Please run data_generator.py first.")
            
        df = pd.read_csv(csv_path)
        
        if intent == "EDA":
            thoughts.append("Executing Exploratory Data Analysis tool...")
            eda_res = run_full_eda(csv_path)
            
            # If aggregation is requested
            if q_type == "aggregation" or "average" in query.lower() or "mean" in query.lower():
                thoughts.append("Aggregating statistics based on query filter...")
                # We expect segments might be loaded in segmented_customers.csv
                seg_csv = "segmented_customers.csv"
                df_to_use = pd.read_csv(seg_csv) if os.path.exists(seg_csv) else df
                
                # Perform group aggregation
                agg_data = {}
                for col in cols:
                    if col in df_to_use.columns and pd.api.types.is_numeric_dtype(df_to_use[col]):
                        if 'segment_id' in df_to_use.columns:
                            agg_data[col] = df_to_use.groupby('segment_id')[col].mean().to_dict()
                        else:
                            agg_data[col] = {"all": float(df_to_use[col].mean())}
                            
                summary_prompt = f"""You are a Banking Data Analyst.
The user asked: "{query}"
We analyzed the dataset and compiled these averages:
{json.dumps(agg_data, indent=2)}

Please write a highly professional, clear response answering the user's question directly.
For reference, here are the segment IDs typically mapped as:
0 = Priority, 1 = Regular, 2 = Dormant, 3 = Overleveraged, 4 = Wealth/Investor (or explain based on cluster indexes).
"""
                resp_text = self._call_ollama(summary_prompt)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'EDA_AGG',
                    'data': agg_data
                }
            
            summary_prompt = f"""You are a Banking Data Analyst.
The user asked: "{query}"
Here is the summary of the Exploratory Data Analysis performed:
- Total customers: {eda_res['total_records']}
- Missing values profile: {json.dumps(eda_res['missing_values'])}
- Statistical ranges: {json.dumps(eda_res['summary_statistics'])}

Write a concise, premium dashboard report outlining customer distribution, data completeness, and key financial patterns. Point out any notable correlations.
"""
            resp_text = self._call_ollama(summary_prompt)
            return {
                'thoughts': thoughts,
                'message': resp_text,
                'intent': 'EDA',
                'data': eda_res
            }
            
        elif intent == "SEGMENTATION":
            thoughts.append("Triggering Feature Engineering pipeline...")
            X, feat_names, pipeline, df_engineered = get_preprocessed_features(df, cols)
            
            thoughts.append(f"Running K-Means clustering (k={num_clusters}) on selected attributes...")
            seg_results = run_kmeans_clustering(df_engineered, X, feat_names, num_clusters=num_clusters)
            
            thoughts.append("Fitting Decision Tree Explainability model to map cluster decision rules...")
            dt_model, importances = fit_explainability_tree(X, seg_results['labels'], feat_names)
            rules = extract_decision_rules(dt_model, feat_names)
            profiles = get_cluster_profiles(seg_results['df'])
            
            # Save segmented dataset to disk
            seg_results['df'].to_csv("segmented_customers.csv", index=False)
            thoughts.append("Saved segmented customer records to 'segmented_customers.csv'.")
            
            # Summarize segments via LLM
            summary_prompt = f"""You are a Retail Banking Marketing Strategist.
We just segmented our customer base into {num_clusters} groups based on these variables: {', '.join(cols)}.
Here are the profiles (average feature values) of the discovered clusters:
{json.dumps(profiles, indent=2)}

Here are the mathematical decision boundaries (Decision Tree rules) for each cluster:
{json.dumps(rules, indent=2)}

Create a compelling name/persona and a detailed description for each segment. 
Profile their typical age, balance, transaction frequency, and credit behaviors.
Present this report in a beautifully formatted table or list.
"""
            resp_text = self._call_ollama(summary_prompt)
            
            return {
                'thoughts': thoughts,
                'message': resp_text,
                'intent': 'SEGMENTATION',
                'data': {
                    'silhouette_score': seg_results['silhouette_score'],
                    'cluster_sizes': seg_results['cluster_sizes'],
                    'profiles': profiles,
                    'rules': rules,
                    'feature_importances': importances
                }
            }
            
        elif intent == "EXPLAIN":
            seg_csv = "segmented_customers.csv"
            if not os.path.exists(seg_csv):
                thoughts.append("No segmented data found. Auto-running default segmentation first...")
                self.execute_query("Segment customers based on balance and transaction frequency", csv_path)
                df_seg = pd.read_csv(seg_csv)
            else:
                df_seg = pd.read_csv(seg_csv)
                
            # Perform explainability tasks
            if q_type == "single_customer" and cust_id:
                thoughts.append(f"Looking up customer record for {cust_id}...")
                cust_row = df_seg[df_seg['customer_id'] == cust_id]
                if cust_row.empty:
                    return {
                        'thoughts': thoughts,
                        'message': f"Customer ID {cust_id} not found in the dataset.",
                        'intent': 'EXPLAIN'
                    }
                
                thoughts.append(f"Tracing decision tree path for customer {cust_id}...")
                # Re-train tree on current segmentation to get trace
                # In real code, we cache this, but re-running is fast enough
                # Extract columns that were clustered
                cols_used = [c for c in df_seg.columns if c not in ['customer_id', 'occupation', 'region', 'segment_id', 'is_outlier', 'is_boundary']]
                X_all, feat_names, pipeline, _ = get_preprocessed_features(df, cols_used)
                dt_model, importances = fit_explainability_tree(X_all, df_seg['segment_id'], feat_names)
                
                explanation = explain_single_customer(cust_row, pipeline, dt_model, feat_names)
                profiles = get_cluster_profiles(df_seg)
                recs = recommend_products_for_customer(cust_row.iloc[0], explanation['assigned_segment'], profiles)
                
                thoughts.append("Synthesizing tailored explainability profile card...")
                explain_prompt = f"""You are a Customer Relationship Officer at a bank.
Explain to a branch manager why Customer {cust_id} was classified into Segment {explanation['assigned_segment']}.

Customer Profile:
{json.dumps(cust_row.iloc[0].to_dict(), indent=2)}

Decision path rules evaluated:
{json.dumps(explanation['path'], indent=2)}

Explain which attributes triggered their segment assignment (e.g. balance, DTI, login habits) in natural, easy-to-understand language. Keep it professional.
"""
                resp_text = self._call_ollama(explain_prompt)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'EXPLAIN_SINGLE',
                    'data': {
                        'customer_id': cust_id,
                        'profile': cust_row.iloc[0].to_dict(),
                        'explanation_path': explanation['path'],
                        'assigned_segment': explanation['assigned_segment'],
                        'recommendations': recs
                    }
                }
            else:
                # Global rules explanation
                thoughts.append("Compiling global segment decision rules...")
                cols_used = [c for c in df_seg.columns if c not in ['customer_id', 'occupation', 'region', 'segment_id', 'is_outlier', 'is_boundary']]
                X_all, feat_names, pipeline, _ = get_preprocessed_features(df, cols_used)
                dt_model, importances = fit_explainability_tree(X_all, df_seg['segment_id'], feat_names)
                rules = extract_decision_rules(dt_model, feat_names)
                
                explain_prompt = f"""You are a Data Science Translator.
Translate these mathematical decision tree rules defining our customer segments into clear, business-friendly rule criteria.
Rules data:
{json.dumps(rules, indent=2)}

For each segment, write a list of logical criteria (e.g., "Priority Segment: Customer must have balance > $45k AND monthly transactions > 12").
"""
                resp_text = self._call_ollama(explain_prompt)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'EXPLAIN_GLOBAL',
                    'data': {
                        'rules': rules,
                        'feature_importances': importances
                    }
                }
                
        elif intent == "RECOMMEND":
            seg_csv = "segmented_customers.csv"
            if not os.path.exists(seg_csv):
                thoughts.append("No segmented data found. Auto-running default segmentation first...")
                self.execute_query("Segment customers based on balance and transaction frequency", csv_path)
                df_seg = pd.read_csv(seg_csv)
            else:
                df_seg = pd.read_csv(seg_csv)
                
            profiles = get_cluster_profiles(df_seg)
            
            if q_type == "single_customer" and cust_id:
                thoughts.append(f"Retrieving profile and computing cross-sell targets for {cust_id}...")
                cust_row = df_seg[df_seg['customer_id'] == cust_id]
                if cust_row.empty:
                    return {
                        'thoughts': thoughts,
                        'message': f"Customer ID {cust_id} not found.",
                        'intent': 'RECOMMEND'
                    }
                
                row_data = cust_row.iloc[0]
                segment = int(row_data['segment_id'])
                recs = recommend_products_for_customer(row_data, segment, profiles)
                
                # If they are eligible to upgrade to Priority (Segment 0)
                transition_steps = []
                if segment != 0:
                    thoughts.append(f"Calculating segment conversion gap between {cust_id} and Priority Average...")
                    transition_steps = calculate_transition_steps(row_data, 0, profiles)
                    
                recs_prompt = f"""You are a Personalized Banking Campaign Planner.
Draft a highly personalized, friendly email offer for Customer {cust_id} offering them financial services.

Customer Profile:
{json.dumps(row_data.to_dict(), indent=2)}

Recommended Products based on rules:
{json.dumps(recs, indent=2)}

Upgrading/Transition steps needed to become a Priority member:
{json.dumps(transition_steps, indent=2)}

Make the email sound extremely premium, supportive, and provide actionable tips on how they can transition to Priority status.
"""
                resp_text = self._call_ollama(recs_prompt)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'RECOMMEND_SINGLE',
                    'data': {
                        'customer_id': cust_id,
                        'recommendations': recs,
                        'transition_steps': transition_steps
                    }
                }
            else:
                # Global recommendations or conversion list (Which regular customers can be converted?)
                thoughts.append("Analyzing regular customers with transition potential...")
                # Filter Regular customers (Segment 1) who have high credit score and are close to Priority (Segment 0) boundaries
                # Let's say: balance > 30000 (Priority mean is around 85000, regular is 15000) or transaction_frequency > 15
                regular_customers = df_seg[df_seg['segment_id'] == 1]
                priority_profile = profiles.get(0, {})
                
                # Score them on closeness: high balance or high transaction frequency
                target_bal = priority_profile.get('account_balance', 50000)
                target_freq = priority_profile.get('transaction_frequency', 15)
                
                # Close candidates: balance > target * 0.6 OR transaction frequency > target * 0.8
                candidates = regular_customers[
                    (regular_customers['account_balance'] >= target_bal * 0.6) | 
                    (regular_customers['transaction_frequency'] >= target_freq * 0.7)
                ]
                
                # Sample top 10 candidates sorted by balance
                top_candidates = candidates.sort_values(by='account_balance', ascending=False).head(10)
                candidates_list = top_candidates[['customer_id', 'account_balance', 'transaction_frequency', 'annual_income', 'credit_score']].to_dict(orient='records')
                
                thoughts.append(f"Identified {len(candidates)} transition candidates. Selected top 10 for the report.")
                
                recs_prompt = f"""You are a Marketing Lead at a retail bank.
We are running a campaign to convert high-performing "Regular" customers into "Priority" customers.
Here are the top 10 conversion candidates:
{json.dumps(candidates_list, indent=2)}

Write an executive briefing detailing:
1. What distinguishes these regular customers from the average regular customer.
2. What specific marketing action should be taken to encourage them to upgrade their balances and transaction activity (e.g. customized interest rates, waiving card fees, personal relationship manager outreach).
3. The average metrics of regular vs priority customers for reference.
"""
                resp_text = self._call_ollama(recs_prompt)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'RECOMMEND_GLOBAL',
                    'data': {
                        'candidates': candidates_list,
                        'total_eligible_candidates': len(candidates)
                    }
                }
                
        else:
            raise ValueError(f"Unknown intent: {intent}")
            
    def close(self):
        self.client.close()
