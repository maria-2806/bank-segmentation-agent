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
from agent_backend.evaluation_tool import evaluate_cluster_range, evaluate_explainability_tree

class AgentOrchestrator:
    def __init__(self, ollama_host="http://localhost:11434", model_name="llama3.2"):
        self.ollama_host = ollama_host.rstrip('/')
        self.model_name = model_name
        self.client = httpx.Client(timeout=300.0) # 5 minute timeout for local CPU inference
        self.pending = None  # holds an unanswered clarifying question (human-in-the-loop)
        
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
            
    def _call_ollama(self, prompt, format_json=False, max_tokens=None):
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3}
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
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
  "num_clusters": null or integer (if segmentation or grouping is requested, default to 5 if not specified),
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
            plan = {"intent": "EDA", "columns": [], "num_clusters": 5, "customer_id": None, "question_type": "global"}
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

    # ---- Human-in-the-loop helpers -------------------------------------------------
    _FEATURE_WORDS = ['balance', 'transaction', 'transactions', 'spend', 'spending',
                      'income', 'credit', 'score', 'age', 'tenure', 'login', 'logins',
                      'investment', 'debt', 'utilization', 'frequency', 'activity']

    _COLMAP = {
        'balance': 'account_balance', 'transaction': 'transaction_frequency',
        'transactions': 'transaction_frequency', 'frequency': 'transaction_frequency',
        'activity': 'transaction_frequency', 'spend': 'avg_transaction_amount',
        'spending': 'avg_transaction_amount', 'amount': 'avg_transaction_amount',
        'income': 'annual_income', 'credit': 'credit_score', 'score': 'credit_score',
        'age': 'age', 'tenure': 'tenure_months', 'login': 'online_login_frequency',
        'logins': 'online_login_frequency', 'investment': 'investment_balance',
        'debt': 'debt_to_income', 'utilization': 'credit_card_utilization',
    }

    def _needs_clarification(self, query, intent, cust_id, q_type):
        """
        Returns a clarifying question when a query is too under-specified to act on,
        or None to proceed. Deliberately conservative so detailed queries run straight
        through and only genuinely vague ones trigger a question.
        """
        q = query.lower()
        if intent == "SEGMENTATION" and not any(w in q for w in self._FEATURE_WORDS):
            return {
                'missing': ['segmentation attributes'],
                'question': ("I can segment the customer base, but I need to know what to segment on. "
                             "Which attributes should I use — e.g. account balance, transaction frequency, "
                             "annual income, or credit score — and how many segments would you like? "
                             "(Default is 5.)"),
                'options': ['account_balance', 'transaction_frequency', 'annual_income', 'credit_score'],
            }
        if intent in ("EXPLAIN", "RECOMMEND") and q_type == "single_customer" and not cust_id:
            return {
                'missing': ['customer id'],
                'question': "Which customer should I look at? Please provide the ID (e.g. CUST0104).",
            }
        return None

    def _apply_clarification(self, pending, answer):
        """Merge the user's answer to a clarifying question back into the original plan."""
        import re
        intent = pending['intent']
        q_type = pending.get('q_type', 'global')
        cust_id = pending.get('cust_id')
        a = answer.lower()

        cols = []
        for word, col in self._COLMAP.items():
            if word in a and col not in cols:
                cols.append(col)
        if not cols:
            cols = ['account_balance', 'transaction_frequency', 'annual_income',
                    'credit_score', 'tenure_months']

        num_clusters = 5
        mnum = re.search(r'\b(\d+)\b', a)
        if mnum and 2 <= int(mnum.group(1)) <= 10:
            num_clusters = int(mnum.group(1))
        else:
            for word, n in {'two': 2, 'three': 3, 'four': 4, 'five': 5,
                            'six': 6, 'seven': 7, 'eight': 8}.items():
                if word in a:
                    num_clusters = n
                    break

        mid = re.search(r'CUST\d+', answer.upper())
        if mid:
            cust_id = mid.group(0)
            q_type = 'single_customer'

        return intent, cols, num_clusters, cust_id, q_type

    def execute_query(self, query, csv_path="banking_customers.csv"):
        thoughts = []
        import re

        if self.pending:
            # This message answers our previous clarifying question — merge and proceed
            thoughts.append("Incorporating your clarification into the original request...")
            intent, cols, num_clusters, cust_id, q_type = self._apply_clarification(self.pending, query)
            self.pending = None
        else:
            thoughts.append("Consulting Ollama LLM to parse query intent...")

            try:
                plan = self.parse_query(query)
            except Exception as e:
                raise RuntimeError(f"Ollama Connection Failed. Please ensure Ollama is running and has model {self.model_name}. Details: {e}")

            raw_intent = str(plan.get("intent", "EDA")).upper()
            if "RECOMMEND" in raw_intent:
                intent = "RECOMMEND"
            elif "EXPLAIN" in raw_intent:
                intent = "EXPLAIN"
            elif "SEGMENT" in raw_intent:
                intent = "SEGMENTATION"
            else:
                intent = "EDA"

            cols = plan.get("columns", [])
            num_clusters = plan.get("num_clusters") or 5
            cust_id = plan.get("customer_id")
            q_type = plan.get("question_type", "global")

            # Force deterministic regex detection of customer ID and intent if present in query string
            match = re.search(r'CUST\d+', query.upper())
            if match:
                cust_id = match.group(0)
                q_type = "single_customer"
                q_low = query.lower()
                if any(k in q_low for k in ["recommend", "product", "offer", "suggest", "proposal", "card"]):
                    intent = "RECOMMEND"
                elif any(k in q_low for k in ["explain", "why", "detail", "profile", "reason", "trace"]):
                    intent = "EXPLAIN"

            # --- Human-in-the-loop gate: ask before guessing when key inputs are missing ---
            clar = self._needs_clarification(query, intent, cust_id, q_type)
            if clar:
                self.pending = {
                    'intent': intent, 'cust_id': cust_id, 'q_type': q_type,
                    'original_query': query, 'missing': clar['missing'],
                }
                thoughts.append(f"Query under-specified ({', '.join(clar['missing'])}) — asking the user.")
                return {
                    'thoughts': thoughts,
                    'message': clar['question'],
                    'question': clar['question'],
                    'options': clar.get('options', []),
                    'intent': 'CLARIFY',
                    'needs_input': True,
                }

            # Ensure we have column defaults (only reached when not clarifying)
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
                            
                summary_prompt = f"""You are a banking data analyst. The user asked: "{query}"
Computed averages (by segment):
{json.dumps(agg_data, indent=2)}

Answer the question directly in under 80 words. Lead with the numbers. No preamble, no marketing language. Segment IDs: 0=Priority, 1=Regular, 2=Dormant, 3=Overleveraged, 4=Wealth/Investor.
"""
                resp_text = self._call_ollama(summary_prompt, max_tokens=220)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'EDA_AGG',
                    'data': agg_data
                }
            
            summary_prompt = f"""You are a banking data analyst. The user asked: "{query}"
EDA results:
- Total customers: {eda_res['total_records']}
- Missing values: {json.dumps(eda_res['missing_values'])}
- Statistics: {json.dumps(eda_res['summary_statistics'])}

Summarize in under 100 words: customer distribution, data completeness, and any notable correlation. Use at most 4 short bullets. No preamble.
"""
            resp_text = self._call_ollama(summary_prompt, max_tokens=280)
            return {
                'thoughts': thoughts,
                'message': resp_text,
                'intent': 'EDA',
                'data': eda_res
            }
            
        elif intent == "SEGMENTATION":
            thoughts.append("Triggering Feature Engineering pipeline...")
            # Persona labels (Priority/Regular/Dormant/Overleveraged/Wealth) are defined over
            # the full behavioural profile, so we always cluster on that complete set — otherwise
            # clustering on 1-2 columns yields value *bands* that get mislabelled as personas.
            # The user's requested columns are still surfaced in the narrative as the focus.
            PERSONA_FEATURES = ['account_balance', 'transaction_frequency', 'annual_income',
                                'credit_score', 'debt_to_income', 'credit_card_utilization',
                                'online_login_frequency', 'investment_balance']
            requested_cols = cols
            cluster_cols = list(PERSONA_FEATURES)
            for c in requested_cols:  # include any extra numeric column the user named
                if c not in cluster_cols:
                    cluster_cols.append(c)
            X, feat_names, pipeline, df_engineered = get_preprocessed_features(df, cluster_cols)
            
            thoughts.append(f"Running K-Means clustering (k={num_clusters}) on the full behavioural profile "
                            f"(focus: {', '.join(requested_cols)})...")
            seg_results = run_kmeans_clustering(df_engineered, X, feat_names, num_clusters=num_clusters)
            
            thoughts.append("Fitting Decision Tree Explainability model to map cluster decision rules...")
            dt_model, importances = fit_explainability_tree(X, seg_results['labels'], feat_names)
            scaler = pipeline.named_steps['scaler'] if hasattr(pipeline, 'named_steps') and 'scaler' in pipeline.named_steps else None
            rules = extract_decision_rules(dt_model, feat_names, scaler=scaler)
            profiles = get_cluster_profiles(seg_results['df'])
            # Canonical name per segment id, so the narrative can't contradict the labelled table
            seg_names = seg_results['df'].groupby('segment_id')['segment_name'].first().to_dict()
            
            # Model evaluation: justify k and quantify how well the rules explain clusters
            thoughts.append("Evaluating cluster quality (k-sweep) and decision-rule fidelity...")
            evaluation = {
                'cluster_selection': evaluate_cluster_range(X),
                'rule_fidelity': evaluate_explainability_tree(X, seg_results['labels'])
            }
            
            # Save segmented dataset to disk
            seg_results['df'].to_csv("segmented_customers.csv", index=False)
            thoughts.append("Saved segmented customer records to 'segmented_customers.csv'.")
            
            # Build named, numeric fact lines so the model describes the ACTUAL profile
            profile_facts = []
            for sid in sorted(profiles):
                p = profiles[sid]
                profile_facts.append(
                    f"{seg_names.get(sid, f'Segment {sid}')}: avg balance ${p.get('account_balance',0):,.0f}, "
                    f"{p.get('transaction_frequency',0):.0f} txns/mo, income ${p.get('annual_income',0):,.0f}, "
                    f"credit {p.get('credit_score',0):.0f}, DTI {p.get('debt_to_income',0):.2f}"
                )
            profile_facts = "\n".join(profile_facts)

            summary_prompt = f"""You are a retail banking strategist. These segments are already named and profiled:
{profile_facts}

For each segment, output ONE line: "<the given name> — 6-8 word descriptor". The descriptor MUST be consistent with that segment's numbers above (e.g. a low-balance, low-activity group is dormant, not affluent). Do not rename segments, do not invent numbers, no intro or closing.
"""
            resp_text = self._call_ollama(summary_prompt, max_tokens=220)
            
            # Frontend expects an array of {feature, importance}, sorted high-to-low
            importances_list = [
                {'feature': f, 'importance': float(v)}
                for f, v in sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
            ]
            
            return {
                'thoughts': thoughts,
                'message': resp_text,
                'intent': 'SEGMENTATION',
                'data': {
                    'silhouette_score': seg_results['silhouette_score'],
                    'cluster_sizes': seg_results['cluster_sizes'],
                    'profiles': profiles,
                    'rules': rules,
                    'feature_importances': importances_list,
                    'evaluation': evaluation
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
                cust_row = df_seg[df_seg['customer_id'].astype(str).str.upper() == str(cust_id).upper()]
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
                # Compare the customer to their segment's averages so wording is relative, not guessed
                assigned = explanation['assigned_segment']
                seg_profile = profiles.get(assigned, {})
                seg_name = cust_row.iloc[0].get('segment_name', f"Segment {assigned}")
                cust = cust_row.iloc[0]
                standing = []
                for key, label in [('account_balance', 'balance'), ('transaction_frequency', 'transactions/mo'),
                                   ('credit_score', 'credit score'), ('debt_to_income', 'debt-to-income'),
                                   ('online_login_frequency', 'logins/mo')]:
                    cv = float(cust.get(key, 0) or 0)
                    sv = float(seg_profile.get(key, 0) or 0)
                    if sv:
                        rel = 'above' if cv > sv * 1.1 else 'below' if cv < sv * 0.9 else 'typical for'
                    else:
                        rel = 'at'
                    standing.append(f"{label}: {cv:,.2f} ({rel} the {seg_name} average of {sv:,.2f})")
                standing_facts = "\n".join(standing)

                explain_prompt = f"""You are a bank relationship officer.
Customer {cust_id} was placed in the "{seg_name}" segment.
How this customer compares to that segment's averages:
{standing_facts}

In under 70 words, name the 2-3 attributes that best explain why they fit this segment. Describe values ONLY as above/below/typical for the segment as stated above — never call a value "high" or "low" in absolute terms. Plain language, no preamble.
"""
                resp_text = self._call_ollama(explain_prompt, max_tokens=200)
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
                
                explain_prompt = f"""You are a data science translator.
Convert these decision-tree rules into business criteria.
Rules: {json.dumps(rules, indent=2)}

For each segment output ONE line: "Name: rule (e.g. balance > $45k AND transactions > 12)". No intro or closing.
"""
                resp_text = self._call_ollama(explain_prompt, max_tokens=250)
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
                cust_row = df_seg[df_seg['customer_id'].astype(str).str.upper() == str(cust_id).upper()]
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
                    
                recs_prompt = f"""You are a banking campaign planner. Write a SHORT offer to Customer {cust_id}.
Profile: {json.dumps(row_data.to_dict(), indent=2)}
Recommended products: {json.dumps(recs, indent=2)}
Steps to reach Priority: {json.dumps(transition_steps, indent=2)}

Under 110 words: a one-line greeting, 1-2 product suggestions, and 2 concrete steps to reach Priority. Warm but direct. No filler.
"""
                resp_text = self._call_ollama(recs_prompt, max_tokens=300)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'RECOMMEND_SINGLE',
                    'data': {
                        'customer_id': cust_id,
                        'profile': row_data.to_dict(),
                        'assigned_segment': segment,
                        'explanation_path': [],
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
                
                # Compute the REAL Regular-vs-Priority gaps so the narrative reports facts, not guesses
                regular_profile = profiles.get(1, {})
                gap_metrics = [
                    ('transaction_frequency', 'monthly transactions', '{:.1f}'),
                    ('annual_income', 'annual income', '${:,.0f}'),
                    ('account_balance', 'account balance', '${:,.0f}'),
                    ('credit_score', 'credit score', '{:.0f}'),
                ]
                gaps = {}
                for key, label, fmt in gap_metrics:
                    reg = float(regular_profile.get(key, 0) or 0)
                    pri = float(priority_profile.get(key, 0) or 0)
                    pct = ((pri - reg) / reg * 100) if reg else 0.0
                    gaps[key] = {
                        'label': label,
                        'regular_avg': round(reg, 2),
                        'priority_avg': round(pri, 2),
                        'absolute_gap': round(pri - reg, 2),
                        'percent_gap': round(pct, 1),
                        'summary': f"{label}: Regular avg {fmt.format(reg)} vs Priority avg {fmt.format(pri)} "
                                   f"({pct:+.0f}% gap)",
                    }
                gap_facts = "\n".join(g['summary'] for g in gaps.values())
                
                recs_prompt = f"""You are a retail bank marketing lead.
Top 10 "Regular" customers with potential to become "Priority":
{json.dumps(candidates_list, indent=2)}

COMPUTED Regular-vs-Priority gaps (use these EXACT numbers, do not invent any figures):
{gap_facts}

In under 130 words and 3 short bullets: (1) what sets these candidates apart, (2) the specific action to convert them, (3) the key metric gaps vs. Priority — quote only the computed numbers above. No preamble.
"""
                resp_text = self._call_ollama(recs_prompt, max_tokens=320)
                return {
                    'thoughts': thoughts,
                    'message': resp_text,
                    'intent': 'RECOMMEND_GLOBAL',
                    'data': {
                        'candidates': candidates_list,
                        'total_eligible_candidates': len(candidates),
                        'conversion_gaps': gaps
                    }
                }
                
        else:
            raise ValueError(f"Unknown intent: {intent}")
            
    def close(self):
        self.client.close()
