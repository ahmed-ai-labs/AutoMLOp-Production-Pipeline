"""
Feature engineering and transformation utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering utilities for creating and transforming features."""
    
    def __init__(self):
        self.fitted_transformers = {}
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[tuple] = None) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        df_enhanced = df.copy()
        numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        if feature_pairs is None:
            # Create interactions for all pairs (limited to first 5 features to avoid explosion)
            feature_pairs = [(numeric_cols[i], numeric_cols[j]) 
                           for i in range(min(5, len(numeric_cols))) 
                           for j in range(i+1, min(5, len(numeric_cols)))]
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df_enhanced.columns and feat2 in df_enhanced.columns:
                # Multiplication interaction
                df_enhanced[f"{feat1}_x_{feat2}"] = df_enhanced[feat1] * df_enhanced[feat2]
                
                # Addition interaction
                df_enhanced[f"{feat1}_plus_{feat2}"] = df_enhanced[feat1] + df_enhanced[feat2]
                
                # Division interaction (avoid division by zero)
                df_enhanced[f"{feat1}_div_{feat2}"] = df_enhanced[feat1] / (df_enhanced[feat2] + 1e-8)
        
        logger.info(f"Created interaction features. New shape: {df_enhanced.shape}")
        return df_enhanced
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2, 
                                 include_cols: List[str] = None) -> pd.DataFrame:
        """Create polynomial features for specified columns."""
        df_poly = df.copy()
        numeric_cols = df_poly.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        if include_cols:
            numeric_cols = [col for col in numeric_cols if col in include_cols]
        
        # Limit to first 3 columns to avoid feature explosion
        numeric_cols = numeric_cols[:3]
        
        for col in numeric_cols:
            for d in range(2, degree + 1):
                df_poly[f"{col}_power_{d}"] = df_poly[col] ** d
        
        logger.info(f"Created polynomial features. New shape: {df_poly.shape}")
        return df_poly
    
    def create_aggregation_features(self, df: pd.DataFrame, group_cols: List[str] = None) -> pd.DataFrame:
        """Create aggregation features based on categorical columns."""
        df_agg = df.copy()
        
        if group_cols is None:
            # Find categorical columns
            group_cols = df_agg.select_dtypes(include=['object', 'int64']).columns.tolist()
            if 'target' in group_cols:
                group_cols.remove('target')
        
        numeric_cols = df_agg.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        for group_col in group_cols[:2]:  # Limit to avoid too many features
            if group_col in df_agg.columns:
                for num_col in numeric_cols[:3]:  # Limit numeric columns
                    if num_col in df_agg.columns:
                        # Mean aggregation
                        group_means = df_agg.groupby(group_col)[num_col].mean()
                        df_agg[f"{group_col}_{num_col}_mean"] = df_agg[group_col].map(group_means)
                        
                        # Standard deviation aggregation
                        group_stds = df_agg.groupby(group_col)[num_col].std().fillna(0)
                        df_agg[f"{group_col}_{num_col}_std"] = df_agg[group_col].map(group_stds)
        
        logger.info(f"Created aggregation features. New shape: {df_agg.shape}")
        return df_agg

class FeatureScaler:
    """Feature scaling utilities."""
    
    def __init__(self, method: str = 'standard'):
        self.method = method
        self.scaler = None
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler and transform features."""
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_scaled = X.copy()
        
        if numeric_cols:
            X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
            logger.info(f"Fitted and transformed features using {self.method} scaling")
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted scaler."""
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet. Call fit_transform first.")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_scaled = X.copy()
        
        if numeric_cols:
            X_scaled[numeric_cols] = self.scaler.transform(X[numeric_cols])
            logger.info(f"Transformed features using fitted {self.method} scaler")
        
        return X_scaled

class FeatureSelector:
    """Feature selection utilities."""
    
    def __init__(self, method: str = 'univariate', k: int = 10):
        self.method = method
        self.k = k
        self.selector = None
        self.selected_features = None
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit selector and transform features."""
        if self.method == 'univariate':
            self.selector = SelectKBest(score_func=f_classif, k=min(self.k, X.shape[1]))
            X_selected = pd.DataFrame(
                self.selector.fit_transform(X, y),
                columns=X.columns[self.selector.get_support()],
                index=X.index
            )
            self.selected_features = X.columns[self.selector.get_support()].tolist()
            
        elif self.method == 'correlation':
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            X_selected = X.drop(columns=to_drop)
            self.selected_features = X_selected.columns.tolist()
            
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
        
        logger.info(f"Selected {len(self.selected_features)} features using {self.method} method")
        return X_selected
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using fitted selector."""
        if self.selected_features is None:
            raise ValueError("Selector not fitted yet. Call fit_transform first.")
        
        return X[self.selected_features]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.selector is None or self.selected_features is None:
            raise ValueError("Selector not fitted yet.")
        
        if hasattr(self.selector, 'scores_'):
            return dict(zip(self.selected_features, self.selector.scores_[self.selector.get_support()]))
        else:
            return {}

def create_feature_pipeline(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    """Complete feature engineering pipeline."""
    logger.info("Starting feature engineering pipeline")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Feature engineering
    engineer = FeatureEngineer()
    X_enhanced = engineer.create_interaction_features(X)
    X_enhanced = engineer.create_polynomial_features(X_enhanced, degree=2)
    X_enhanced = engineer.create_aggregation_features(X_enhanced)
    
    # Feature scaling
    scaler = FeatureScaler(method='standard')
    X_scaled = scaler.fit_transform(X_enhanced)
    
    # Feature selection
    selector = FeatureSelector(method='univariate', k=15)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Combine with target
    result_df = X_selected.copy()
    result_df[target_col] = y
    
    logger.info(f"Feature engineering complete. Final shape: {result_df.shape}")
    return result_df
