"""
Data loading and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading utilities for the MLOps pipeline."""
    
    def __init__(self, data_path: str = "data"):
        self.data_path = Path(data_path)
        
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """Load raw data from file."""
        file_path = self.data_path / "raw" / filename
        
        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Generating sample data.")
            return self.generate_sample_data()
            
        try:
            if filename.endswith('.csv'):
                return pd.read_csv(file_path)
            elif filename.endswith('.parquet'):
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def generate_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate sample data for testing."""
        np.random.seed(42)
        
        data = {
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(2, 1.5, n_samples),
            'feature_3': np.random.exponential(1, n_samples),
            'feature_4': np.random.uniform(-1, 1, n_samples),
            'feature_5': np.random.gamma(2, 2, n_samples),
            'categorical_1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical_2': np.random.choice(['X', 'Y', 'Z'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable based on features
        df['target'] = (
            (df['feature_1'] > 0) & 
            (df['feature_2'] > 2) | 
            (df['categorical_1'] == 'A')
        ).astype(int)
        
        logger.info(f"Generated sample data with {n_samples} samples")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed data."""
        file_path = self.data_path / "processed" / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if filename.endswith('.csv'):
            df.to_csv(file_path, index=False)
        elif filename.endswith('.parquet'):
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
            
        logger.info(f"Saved processed data to {file_path}")

class DataPreprocessor:
    """Data preprocessing utilities."""
    
    def __init__(self):
        self.fitted_params = {}
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers."""
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
            
        for col in df_clean.select_dtypes(include=['object']).columns:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Remove outliers using IQR method
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if col != 'target':  # Don't remove outliers from target
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_clean = df_clean[
                    (df_clean[col] >= lower_bound) & 
                    (df_clean[col] <= upper_bound)
                ]
        
        logger.info(f"Data cleaned. Shape: {df_clean.shape}")
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df_encoded = df.copy()
        
        for col in df_encoded.select_dtypes(include=['object']).columns:
            if col != 'target':
                # Simple label encoding for demonstration
                unique_values = df_encoded[col].unique()
                encoding_map = {val: idx for idx, val in enumerate(unique_values)}
                df_encoded[col] = df_encoded[col].map(encoding_map)
                
                # Store encoding map for inverse transformation
                self.fitted_params[f"{col}_encoding"] = encoding_map
        
        logger.info("Categorical variables encoded")
        return df_encoded
    
    def split_features_target(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[pd.DataFrame, pd.Series]:
        """Split features and target variable."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
