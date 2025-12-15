"""
XGBoost Training Pipeline with GPU Support
Robust, auditable ML pipeline with comprehensive logging
"""

import pandas as pd
import numpy as np
import json
import yaml
import joblib
import hashlib
import warnings
import sys
import os
import gc
import re
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from scipy import stats
import argparse

# ML libraries
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')

# Seeds
np.random.seed(42)
random.seed(42)

# Configuration
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

class Logger:
    """Centralized logging system"""
    def __init__(self):
        self.main_log = []
        self.step_logs = {}
        
    def log(self, message: str, step: str = "main"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{step}] {message}"
        print(log_entry)
        
        if step == "main":
            self.main_log.append(log_entry)
        else:
            if step not in self.step_logs:
                self.step_logs[step] = []
            self.step_logs[step].append(log_entry)
    
    def save_logs(self):
        # Main debug log
        with open(ARTIFACTS_DIR / "debug.log", "w", encoding="utf-8") as f:
            f.write("\n".join(self.main_log))
        
        # Step-specific logs
        for step, logs in self.step_logs.items():
            with open(ARTIFACTS_DIR / f"debug_step_{step}.log", "w", encoding="utf-8") as f:
                f.write("\n".join(logs))

logger = Logger()

def save_exception(e: Exception, step: str):
    """Save exception details"""
    import traceback
    with open(ARTIFACTS_DIR / "debug_exception.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Step: {step}\n")
        f.write(f"Exception: {str(e)}\n")
        f.write(f"Traceback:\n{traceback.format_exc()}\n")

def step_0_load_and_assert(csv_path: str) -> pd.DataFrame:
    """STEP 0: Load and validate CSV structure"""
    logger.log("STEP 0: Loading and validating CSV", "01")
    
    try:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
        logger.log(f"Loaded shape: {df.shape}", "01")
        logger.log(f"Columns found: {list(df.columns)}", "01")
        
        required_cols = ['date', 'product_id', 'units_sold', 'sale_price', 
                        'is_promo', 'lag_7', 'product_id_encoded']
        
        if list(df.columns) != required_cols:
            missing = set(required_cols) - set(df.columns)
            extra = set(df.columns) - set(required_cols)
            error_msg = f"Column mismatch!\nMissing: {missing}\nExtra: {extra}"
            logger.log(error_msg, "01")
            raise ValueError(error_msg)
        
        # Save versions
        versions = {
            'python': sys.version,
            'pandas': pd.__version__,
            'numpy': np.__version__,
            'xgboost': xgb.__version__,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(ARTIFACTS_DIR / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.dump(versions, f)
        
        logger.log("✓ STEP 0 Complete", "01")
        return df
        
    except Exception as e:
        save_exception(e, "step_0")
        raise

def step_1_super_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 1: Super cleaner with repair tracking"""
    logger.log("STEP 1: Super cleaning and repairs", "02")
    
    repairs = []
    numeric_cols = ['units_sold', 'sale_price', 'lag_7', 'product_id_encoded', 'is_promo']
    
    try:
        for col in numeric_cols:
            logger.log(f"Cleaning column: {col}", "02")
            original = df[col].copy()
            
            # Trim whitespace
            df[col] = df[col].astype(str).str.strip()
            
            # Check for brackets/arrays
            bracket_pattern = r'^\s*[\[\(].*[\]\)]\s*$'
            bracket_mask = df[col].str.match(bracket_pattern, na=False)
            
            if bracket_mask.any():
                logger.log(f"Found {bracket_mask.sum()} bracketed values in {col}", "02")
                for idx in df[bracket_mask].index:
                    orig_val = df.loc[idx, col]
                    # Extract first numeric
                    match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', orig_val)
                    if match:
                        new_val = match.group(0)
                        df.loc[idx, col] = new_val
                        repairs.append({
                            'row': idx,
                            'column': col,
                            'original': orig_val,
                            'repaired': new_val
                        })
            
            # Remove currency symbols
            df[col] = df[col].str.replace(r'[$€£]', '', regex=True)
            
            # Handle comma/dot locale issues
            has_both = df[col].str.contains(r'\d+[,.]\d+[,.]\d+', na=False)
            if has_both.any():
                logger.log(f"Found locale formatting in {col}", "02")
                # Assume last separator is decimal
                df.loc[has_both, col] = df.loc[has_both, col].str.replace(',', '', regex=False).str.replace('.', '.', regex=False)
            
            # Coerce to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                logger.log(f"Coerced {nan_count} values to NaN in {col}", "02")
        
        # Date cleaning
        logger.log("Cleaning date column", "02")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        nat_count = df['date'].isna().sum()
        
        if nat_count > 0:
            logger.log(f"ERROR: {nat_count} unparseable dates found", "02")
            raise ValueError(f"Date parsing failed for {nat_count} rows")
        
        # Save repairs
        if repairs:
            pd.DataFrame(repairs).to_csv(ARTIFACTS_DIR / "repairs.csv", index=False, encoding="utf-8")
            logger.log(f"Saved {len(repairs)} repairs", "02")
        
        # Save cleaned CSV
        df.to_csv(ARTIFACTS_DIR / "daily_model_ready.cleaned.csv", index=False, encoding="utf-8")
        logger.log("✓ STEP 1 Complete", "02")
        
        return df
        
    except Exception as e:
        save_exception(e, "step_1")
        raise

def step_2_anti_leakage(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """STEP 2: Anti-leakage - drop dangerous columns"""
    logger.log("STEP 2: Anti-leakage check", "03")
    
    try:
        # Extract target
        y = df['units_sold'].copy()
        
        # Check for leakage columns
        leakage_pattern = r'(?i).*sale.*|.*_log.*|.*_log1p.*|.*_per_stock.*|revenue|forecast|demand|orders|units_ordered'
        
        cols_to_check = [c for c in df.columns if c not in ['units_sold', 'sale_price', 'date', 'product_id']]
        dropped = []
        
        for col in cols_to_check:
            if re.match(leakage_pattern, col):
                dropped.append(col)
        
        if dropped:
            logger.log(f"Dropping leakage columns: {dropped}", "03")
            df = df.drop(columns=dropped)
        else:
            logger.log("No leakage columns detected", "03")
        
        logger.log("✓ STEP 2 Complete", "03")
        return df, y
        
    except Exception as e:
        save_exception(e, "step_2")
        raise

def step_3_type_sanity(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 3: Type validation and sanity checks"""
    logger.log("STEP 3: Type and sanity checks", "04")
    
    try:
        # Units sold: integer >= 0
        if df['units_sold'].min() < 0:
            neg_count = (df['units_sold'] < 0).sum()
            if neg_count / len(df) < 0.001:
                logger.log(f"Clipping {neg_count} negative units_sold to 0", "04")
                df.loc[df['units_sold'] < 0, 'units_sold'] = 0
            else:
                raise ValueError(f"Too many negative units_sold: {neg_count}")
        
        df['units_sold'] = df['units_sold'].astype(int)
        
        # Sale price: float >= 0
        if df['sale_price'].min() < 0:
            neg_count = (df['sale_price'] < 0).sum()
            if neg_count / len(df) < 0.001:
                logger.log(f"Clipping {neg_count} negative sale_price to 0", "04")
                df.loc[df['sale_price'] < 0, 'sale_price'] = 0
            else:
                raise ValueError(f"Too many negative sale_price: {neg_count}")
        
        df['sale_price'] = df['sale_price'].astype(float)
        
        # is_promo: binary {0,1}
        unique_promo = df['is_promo'].unique()
        if not set(unique_promo).issubset({0, 1}):
            raise ValueError(f"is_promo contains invalid values: {unique_promo}")
        
        df['is_promo'] = df['is_promo'].astype(int)
        
        # lag_7: no missing
        if df['lag_7'].isna().any():
            raise ValueError(f"lag_7 contains {df['lag_7'].isna().sum()} missing values")
        
        df['lag_7'] = df['lag_7'].astype(float)
        
        # product_id_encoded: integer
        df['product_id_encoded'] = df['product_id_encoded'].astype(int)
        
        logger.log(f"✓ All type checks passed. Shape: {df.shape}", "04")
        logger.log("✓ STEP 3 Complete", "04")
        
        return df
        
    except Exception as e:
        save_exception(e, "step_3")
        raise

def step_4_encoder_features(df: pd.DataFrame) -> Dict:
    """STEP 4: Validate encoder and save feature names"""
    logger.log("STEP 4: Encoder validation", "05")
    
    try:
        # Validate product_id_encoded
        product_mapping = df.groupby('product_id')['product_id_encoded'].first().sort_values()
        
        expected_encoding = {pid: idx for idx, pid in enumerate(sorted(df['product_id'].unique()))}
        
        # Save mapping
        with open(ARTIFACTS_DIR / "label_encoder.json", "w", encoding="utf-8") as f:
            json.dump(expected_encoding, f, indent=2)
        
        # Save feature names
        feature_names = ['lag_7', 'sale_price', 'is_promo', 'product_id_encoded']
        with open(ARTIFACTS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feature_names, f, indent=2)
        
        logger.log(f"Feature names: {feature_names}", "05")
        logger.log("✓ STEP 4 Complete", "05")
        
        return {'feature_names': feature_names, 'label_encoder': expected_encoding}
        
    except Exception as e:
        save_exception(e, "step_4")
        raise

def step_5_time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """STEP 5: Time-based train/val/test split"""
    logger.log("STEP 5: Time-based split", "06")
    
    try:
        # Sort by product and date
        df = df.sort_values(['product_id', 'date']).reset_index(drop=True)
        
        # Get unique sorted dates
        unique_dates = sorted(df['date'].unique())
        n_dates = len(unique_dates)
        
        train_cutoff_idx = int(n_dates * 0.70)
        val_cutoff_idx = int(n_dates * 0.85)
        
        train_dates = unique_dates[:train_cutoff_idx]
        val_dates = unique_dates[train_cutoff_idx:val_cutoff_idx]
        test_dates = unique_dates[val_cutoff_idx:]
        
        train = df[df['date'].isin(train_dates)].copy()
        val = df[df['date'].isin(val_dates)].copy()
        test = df[df['date'].isin(test_dates)].copy()
        
        # Assert chronological order
        assert train['date'].max() < val['date'].min(), "Train/Val overlap!"
        assert val['date'].max() < test['date'].min(), "Val/Test overlap!"
        
        logger.log(f"Train: {len(train)} rows, dates {train['date'].min()} to {train['date'].max()}", "06")
        logger.log(f"Val: {len(val)} rows, dates {val['date'].min()} to {val['date'].max()}", "06")
        logger.log(f"Test: {len(test)} rows, dates {test['date'].min()} to {test['date'].max()}", "06")
        
        # Save splits
        train.to_csv(ARTIFACTS_DIR / "train.csv", index=False, encoding="utf-8")
        val.to_csv(ARTIFACTS_DIR / "val.csv", index=False, encoding="utf-8")
        test.to_csv(ARTIFACTS_DIR / "test.csv", index=False, encoding="utf-8")
        
        logger.log("✓ STEP 5 Complete", "06")
        
        return train, val, test
        
    except Exception as e:
        save_exception(e, "step_5")
        raise

def step_6_preprocessing(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Dict:
    """STEP 6: Minimal preprocessing"""
    logger.log("STEP 6: Preprocessing", "07")
    
    try:
        feature_cols = ['lag_7', 'sale_price', 'is_promo', 'product_id_encoded']
        
        X_train = train[feature_cols].copy()
        y_train = train['units_sold'].copy()
        
        X_val = val[feature_cols].copy()
        y_val = val['units_sold'].copy()
        
        X_test = test[feature_cols].copy()
        y_test = test['units_sold'].copy()
        
        # Cast dtypes
        X_train['lag_7'] = X_train['lag_7'].astype(np.float32)
        X_train['sale_price'] = X_train['sale_price'].astype(np.float32)
        X_train['is_promo'] = X_train['is_promo'].astype(np.int8)
        X_train['product_id_encoded'] = X_train['product_id_encoded'].astype(np.int32)
        
        X_val['lag_7'] = X_val['lag_7'].astype(np.float32)
        X_val['sale_price'] = X_val['sale_price'].astype(np.float32)
        X_val['is_promo'] = X_val['is_promo'].astype(np.int8)
        X_val['product_id_encoded'] = X_val['product_id_encoded'].astype(np.int32)
        
        X_test['lag_7'] = X_test['lag_7'].astype(np.float32)
        X_test['sale_price'] = X_test['sale_price'].astype(np.float32)
        X_test['is_promo'] = X_test['is_promo'].astype(np.int8)
        X_test['product_id_encoded'] = X_test['product_id_encoded'].astype(np.int32)
        
        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        # Save preprocessor metadata
        preprocessor = {
            'dtypes': {
                'lag_7': 'float32',
                'sale_price': 'float32',
                'is_promo': 'int8',
                'product_id_encoded': 'int32'
            },
            'feature_names': feature_cols
        }
        
        joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
        
        logger.log(f"X_train shape: {X_train.shape}", "07")
        logger.log(f"X_val shape: {X_val.shape}", "07")
        logger.log(f"X_test shape: {X_test.shape}", "07")
        logger.log("✓ STEP 6 Complete", "07")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'train_dates': train['date'],
            'val_dates': val['date'],
            'test_dates': test['date'],
            'train_products': train['product_id'],
            'val_products': val['product_id'],
            'test_products': test['product_id']
        }
        
    except Exception as e:
        save_exception(e, "step_6")
        raise

def step_7_model_training(data: Dict, enable_gpu: bool = True) -> Dict:
    """STEP 7: XGBoost model training with GPU support"""
    logger.log("STEP 7: Model training", "08")
    
    try:
        # Detect GPU
        gpu_available = False
        if enable_gpu:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True)
                gpu_available = result.returncode == 0
                logger.log(f"GPU available: {gpu_available}", "08")
            except:
                logger.log("GPU check failed, using CPU", "08")
        
        # Set parameters
        params = {
            'objective': 'reg:squarederror',
            'tree_method': 'gpu_hist' if gpu_available else 'hist',
            'device': 'cuda' if gpu_available else 'cpu',
            'seed': 42,
            'eval_metric': 'rmse',
            'verbosity': 1,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 500
        }
        
        logger.log(f"Training with params: {params}", "08")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(data['X_train'], label=data['y_train'])
        dval = xgb.DMatrix(data['X_val'], label=data['y_val'])
        dtest = xgb.DMatrix(data['X_test'], label=data['y_test'])
        
        # Train
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=500,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        # Save model
        bst.save_model(str(ARTIFACTS_DIR / "xgb_model.json"))
        joblib.dump(bst, ARTIFACTS_DIR / "xgb_model.pkl")
        
        logger.log(f"Best iteration: {bst.best_iteration}", "08")
        logger.log("✓ STEP 7 Complete", "08")
        
        return {
            'model': bst,
            'params': params,
            'best_iteration': bst.best_iteration
        }
        
    except Exception as e:
        save_exception(e, "step_7")
        raise

def step_8_metrics_and_checks(model: Dict, data: Dict) -> Dict:
    """STEP 8: Compute metrics and sanity checks"""
    logger.log("STEP 8: Metrics and sanity checks", "09")
    
    try:
        bst = model['model']
        
        # Predictions
        dval = xgb.DMatrix(data['X_val'])
        dtest = xgb.DMatrix(data['X_test'])
        
        pred_val = bst.predict(dval)
        pred_test = bst.predict(dtest)
        
        # Metrics
        val_mae = mean_absolute_error(data['y_val'], pred_val)
        val_rmse = np.sqrt(mean_squared_error(data['y_val'], pred_val))
        
        test_mae = mean_absolute_error(data['y_test'], pred_test)
        test_rmse = np.sqrt(mean_squared_error(data['y_test'], pred_test))
        
        logger.log(f"Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}", "09")
        logger.log(f"Test MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}", "09")
        
        # Range ratio check
        pred_range = pred_test.max() - pred_test.min()
        true_range = data['y_test'].max() - data['y_test'].min()
        range_ratio = pred_range / true_range if true_range > 0 else 0
        
        logger.log(f"Range ratio: {range_ratio:.4f}", "09")
        
        status = "PASS"
        if range_ratio < 0.1:
            logger.log("WARNING: Range ratio < 0.1, possible range collapse", "09")
            status = "WARNING"
        
        # KS test
        ks_stat, ks_pval = stats.ks_2samp(data['y_train'], data['y_val'])
        logger.log(f"KS test: stat={ks_stat:.4f}, p-value={ks_pval:.4f}", "09")
        
        # Save predictions
        predictions = pd.DataFrame({
            'date': data['test_dates'].values,
            'product_id': data['test_products'].values,
            'true_units': data['y_test'].values,
            'predicted_units': pred_test
        })
        predictions.to_csv(ARTIFACTS_DIR / "predictions.csv", index=False, encoding="utf-8")
        
        logger.log("✓ STEP 8 Complete", "09")
        
        return {
            'status': status,
            'val_mae': float(val_mae),
            'val_rmse': float(val_rmse),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'range_ratio': float(range_ratio),
            'ks_stat': float(ks_stat),
            'ks_pval': float(ks_pval)
        }
        
    except Exception as e:
        save_exception(e, "step_8")
        raise

def step_10_explainability(model: Dict, data: Dict) -> Dict:
    """STEP 10: Model explainability"""
    logger.log("STEP 10: Explainability", "10")
    
    try:
        bst = model['model']
        
        # Get feature importance from booster
        importance = bst.get_score(importance_type='gain')
        
        logger.log(f"Feature importance: {importance}", "10")
        
        # Save
        with open(ARTIFACTS_DIR / "perm_importance.json", "w", encoding="utf-8") as f:
            json.dump(importance, f, indent=2)
        
        logger.log("✓ STEP 10 Complete", "10")
        
        return {'feature_importance': importance}
        
    except Exception as e:
        save_exception(e, "step_10")
        return {}

def step_11_final_artifacts(model: Dict, metrics: Dict, explainability: Dict, config: Dict):
    """STEP 11: Save final artifacts and summary"""
    logger.log("STEP 11: Final artifacts", "11")
    
    try:
        # Training summary
        summary = {
            'status': metrics['status'],
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'val_mae': metrics['val_mae'],
                'val_rmse': metrics['val_rmse'],
                'test_mae': metrics['test_mae'],
                'test_rmse': metrics['test_rmse'],
                'range_ratio': metrics['range_ratio']
            },
            'model': {
                'path': 'artifacts/xgb_model.json',
                'best_iteration': model['best_iteration'],
                'params': model['params']
            },
            'feature_importance': explainability.get('feature_importance', {}),
            'config': config
        }
        
        with open(ARTIFACTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        
        # Config hash
        with open(ARTIFACTS_DIR / "config_used.yaml", "r", encoding="utf-8") as f:
            config_content = f.read()
        
        config_hash = hashlib.sha256(config_content.encode()).hexdigest()
        
        with open(ARTIFACTS_DIR / "config_hash.txt", "w", encoding="utf-8") as f:
            f.write(config_hash)
        
        logger.log(f"Config hash: {config_hash}", "11")
        logger.log("✓ STEP 11 Complete", "11")
        
        return summary
        
    except Exception as e:
        save_exception(e, "step_11")
        raise

def main():
    parser = argparse.ArgumentParser(description='XGBoost Training Pipeline')
    parser.add_argument('--csv', type=str, default='daily_model_ready.csv', help='Input CSV path')
    parser.add_argument('--enable-optuna', action='store_true', help='Enable Optuna tuning')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    args = parser.parse_args()
    
    try:
        logger.log("=" * 80, "main")
        logger.log("XGBoost Training Pipeline Starting", "main")
        logger.log("=" * 80, "main")
        
        # Step 0
        df = step_0_load_and_assert(args.csv)
        
        # Step 1
        df = step_1_super_cleaner(df)
        
        # Step 2
        df, y = step_2_anti_leakage(df)
        
        # Step 3
        df = step_3_type_sanity(df)
        
        # Check minimum rows
        if len(df) < 500:
            raise ValueError(f"Insufficient rows after cleaning: {len(df)}")
        
        # Step 4
        encoder_info = step_4_encoder_features(df)
        
        # Step 5
        train, val, test = step_5_time_split(df)
        
        # Step 6
        data = step_6_preprocessing(train, val, test)
        
        # Step 7
        model = step_7_model_training(data, enable_gpu=not args.no_gpu)
        
        # Step 8
        metrics = step_8_metrics_and_checks(model, data)
        
        # Step 10
        explainability = step_10_explainability(model, data)
        
        # Step 11
        summary = step_11_final_artifacts(model, metrics, explainability, encoder_info)
        
        # Save logs
        logger.save_logs()
        
        # Print summary
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Status: {summary['status']}")
        print(f"Val MAE: {summary['metrics']['val_mae']:.4f}")
        print(f"Val RMSE: {summary['metrics']['val_rmse']:.4f}")
        print(f"Test MAE: {summary['metrics']['test_mae']:.4f}")
        print(f"Test RMSE: {summary['metrics']['test_rmse']:.4f}")
        print(f"Range Ratio: {summary['metrics']['range_ratio']:.4f}")
        print(f"Model saved: {summary['model']['path']}")
        print("=" * 80)
        
        if summary['status'] != "PASS":
            print("⚠️  WARNING: Check artifacts/debug.log for details")
        
    except Exception as e:
        logger.log(f"FATAL ERROR: {str(e)}", "main")
        logger.save_logs()
        
        # Save failure summary
        failure_summary = {
            'status': 'FAIL',
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'remediation': 'Check artifacts/debug_exception.log for details'
        }
        
        with open(ARTIFACTS_DIR / "training_summary.json", "w", encoding="utf-8") as f:
            json.dump(failure_summary, f, indent=2)
        
        print("\n" + "=" * 80)
        print("TRAINING FAILED")
        print("=" * 80)
        print(f"Error: {str(e)}")
        print("Check artifacts/debug_exception.log for full traceback")
        print("=" * 80)
        
        raise

if __name__ == "__main__":
    main()