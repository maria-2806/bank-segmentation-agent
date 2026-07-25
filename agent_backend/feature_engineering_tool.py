import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def engineer_features(df):
    """
    Creates derived features that enhance clustering:
    - balance_to_income_ratio
    - cc_utilization_to_limit (derived risk proxy)
    - investment_ratio (investment_balance / total assets)
    """
    df_engineered = df.copy()
    
    # Avoid division by zero
    df_engineered['balance_to_income_ratio'] = df_engineered['account_balance'] / (df_engineered['annual_income'] + 1.0)
    
    total_assets = df_engineered['account_balance'] + df_engineered['investment_balance'] + 1.0
    df_engineered['investment_ratio'] = df_engineered['investment_balance'] / total_assets
    
    return df_engineered

class FeaturePreprocessingPipeline:
    def __init__(self, numeric_features, categorical_features, scaling_method='standard'):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = []
        self.is_fitted = False
        
    def fit_transform(self, df):
        # Fit numeric scaler
        scaled_numeric = self.scaler.fit_transform(df[self.numeric_features])
        
        # Fit categorical encoder
        encoded_categorical = np.array([[]] * len(df))
        categorical_cols_out = []
        
        if self.categorical_features:
            encoded_categorical = self.encoder.fit_transform(df[self.categorical_features])
            categorical_cols_out = list(self.encoder.get_feature_names_out(self.categorical_features))
            
        # Combine
        if self.categorical_features:
            X = np.hstack([scaled_numeric, encoded_categorical])
        else:
            X = scaled_numeric
            
        self.feature_names = self.numeric_features + categorical_cols_out
        self.is_fitted = True
        return X, self.feature_names
        
    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Pipeline is not fitted yet.")
            
        scaled_numeric = self.scaler.transform(df[self.numeric_features])
        
        if self.categorical_features:
            encoded_categorical = self.encoder.transform(df[self.categorical_features])
            X = np.hstack([scaled_numeric, encoded_categorical])
        else:
            X = scaled_numeric
            
        return X

def get_preprocessed_features(df, features_list, scaling_method='standard'):
    """
    Helper function to pre-process a list of features dynamically from a dataframe
    """
    df_engineered = engineer_features(df)
    
    # Filter features that are numeric vs categorical
    numeric_features = []
    categorical_features = []
    
    for feat in features_list:
        if feat in df_engineered.columns:
            if pd.api.types.is_numeric_dtype(df_engineered[feat]):
                numeric_features.append(feat)
            else:
                categorical_features.append(feat)
                
    pipeline = FeaturePreprocessingPipeline(numeric_features, categorical_features, scaling_method)
    X, feature_names = pipeline.fit_transform(df_engineered)
    
    return X, feature_names, pipeline, df_engineered
