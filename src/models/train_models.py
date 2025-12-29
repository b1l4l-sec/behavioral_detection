
import os
import json
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    
    def __init__(
        self,
        models_dir: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.models_dir = Path(models_dir or './data/models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or self._get_default_config()
        self.scaler = StandardScaler()
        self.trained_models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}
        
        logger.info(f"Model trainer initialized: {self.models_dir}")
    
    def _get_default_config(self) -> Dict:
        return {
            'isolation_forest': {
                'n_estimators': 100,
                'contamination': 0.1,
                'random_state': 42
            },
            'one_class_svm': {
                'kernel': 'rbf',
                'nu': 0.1,
                'gamma': 'auto'
            },
            'lof': {
                'n_neighbors': 20,
                'contamination': 0.1
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
        }
    
    def load_data(
        self,
        data_path: str,
        feature_cols: Optional[List[str]] = None,
        target_col: str = 'label_numeric'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        logger.info(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path)
        
        if feature_cols is None:
            exclude_cols = ['label', 'label_numeric', 'window_start', 'window_end', 'event_count']
            feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else None
        
        logger.info(f"Loaded {len(X)} samples")
        logger.info(f"Features: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        scale: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        logger.info(f"Training data: {len(X_train)}")
        logger.info(f"Test data: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_isolation_forest(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        logger.info("🌲 Training Isolation Forest...")
        
        config = self.config['isolation_forest']
        model = IsolationForest(**config)
        
        start_time = time.time()
        model.fit(X_train)
        train_time = time.time() - start_time
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        y_pred_test_binary = np.where(y_pred_test == -1, 1, 0)
        
        metrics = self._calculate_metrics(y_test, y_pred_test_binary, model_type='unsupervised')
        metrics['train_time'] = train_time
        
        self.trained_models['isolation_forest'] = model
        self.results['isolation_forest'] = metrics
        
        return metrics
    
    def train_one_class_svm(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        logger.info("🔮 Training One-Class SVM...")
        
        config = self.config['one_class_svm']
        model = OneClassSVM(**config)
        
        X_train_normal = X_train[np.random.choice(len(X_train), min(1000, len(X_train)), replace=False)]
        
        start_time = time.time()
        model.fit(X_train_normal)
        train_time = time.time() - start_time
        
        y_pred_test = model.predict(X_test)
        y_pred_test_binary = np.where(y_pred_test == -1, 1, 0)
        
        metrics = self._calculate_metrics(y_test, y_pred_test_binary, model_type='unsupervised')
        metrics['train_time'] = train_time
        
        self.trained_models['one_class_svm'] = model
        self.results['one_class_svm'] = metrics
        
        return metrics
    
    def train_lof(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        logger.info("🔍 Training Local Outlier Factor...")
        
        config = self.config['lof']
        model = LocalOutlierFactor(novelty=True, **config)
        
        X_train_sample = X_train[np.random.choice(len(X_train), min(2000, len(X_train)), replace=False)]
        
        start_time = time.time()
        model.fit(X_train_sample)
        train_time = time.time() - start_time
        
        y_pred_test = model.predict(X_test)
        y_pred_test_binary = np.where(y_pred_test == -1, 1, 0)
        
        metrics = self._calculate_metrics(y_test, y_pred_test_binary, model_type='unsupervised')
        metrics['train_time'] = train_time
        
        self.trained_models['lof'] = model
        self.results['lof'] = metrics
        
        return metrics
    
    def train_random_forest(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        logger.info("🌳 Training Random Forest...")
        
        config = self.config['random_forest']
        model = RandomForestClassifier(**config)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred_test, y_prob=y_prob_test, model_type='supervised')
        metrics['train_time'] = train_time
        metrics['feature_importance'] = dict(zip(
            range(X_train.shape[1]),
            model.feature_importances_
        ))
        
        self.trained_models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        return metrics
    
    def train_xgboost(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available")
            return {}
        
        logger.info("🚀 Training XGBoost...")
        
        config = self.config['xgboost']
        model = xgb.XGBClassifier(**config, use_label_encoder=False, eval_metric='logloss')
        
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred_test = model.predict(X_test)
        y_prob_test = model.predict_proba(X_test)[:, 1]
        
        metrics = self._calculate_metrics(y_test, y_pred_test, y_prob=y_prob_test, model_type='supervised')
        metrics['train_time'] = train_time
        metrics['feature_importance'] = dict(zip(
            range(X_train.shape[1]),
            model.feature_importances_
        ))
        
        self.trained_models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_type: str = 'supervised'
    ) -> Dict:
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'model_type': model_type
        }
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        
        if y_prob is not None:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_prob))
            except:
                metrics['auc'] = 0.0
        
        return metrics
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict]:
        logger.info("=" * 60)
        logger.info("🚀 Starting training of all models")
        logger.info("=" * 60)
        
        self.train_isolation_forest(X_train, X_test, y_test)
        self.train_one_class_svm(X_train, X_test, y_test)
        self.train_lof(X_train, X_test, y_test)
        
        self.train_random_forest(X_train, X_test, y_train, y_test)
        self.train_xgboost(X_train, X_test, y_train, y_test)
        
        return self.results
    
    def save_models(self, suffix: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, model in self.trained_models.items():
            if model is not None:
                filename = f"{name}{suffix}_{timestamp}.joblib"
                filepath = self.models_dir / filename
                joblib.dump(model, filepath)
                logger.info(f"Saved: {filename}")
        
        scaler_path = self.models_dir / f"scaler{suffix}_{timestamp}.joblib"
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved")
        
        results_path = self.models_dir / f"results{suffix}_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved")
    
    def load_model(self, model_name: str, model_path: str):
        model = joblib.load(model_path)
        self.trained_models[model_name] = model
        logger.info(f"Model loaded: {model_name}")
        return model
    
    def compare_models(self) -> pd.DataFrame:
        comparison_data = []
        
        for name, metrics in self.results.items():
            if metrics:
                comparison_data.append({
                    'Model': name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0),
                    'AUC': metrics.get('auc', 0),
                    'FPR': metrics.get('false_positive_rate', 0),
                    'Train Time (s)': metrics.get('train_time', 0)
                })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1', ascending=False)
        
        return df
    
    def print_comparison(self):
        df = self.compare_models()
        
        print("\n" + "=" * 90)
        print("📊 Model Performance Comparison")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)
        
        if len(df) > 0:
            best_model = df.iloc[0]['Model']
            best_f1 = df.iloc[0]['F1']
            print(f"\n🏆 Best Model: {best_model} (F1: {best_f1:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Model Training"
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to data file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data/models',
        help='Directory to save models'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test ratio'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        logger.error(f"File not found: {args.data}")
        return
    
    trainer = ModelTrainer(models_dir=args.output)
    
    X, y, feature_names = trainer.load_data(args.data)
    
    if y is None:
        logger.error("No target column")
        return
    
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, test_size=args.test_size)
    
    trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    trainer.print_comparison()
    
    trainer.save_models()
    
    logger.info("✅ Finished")


if __name__ == "__main__":
    main()
