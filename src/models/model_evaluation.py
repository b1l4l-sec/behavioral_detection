
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import logging
import joblib

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


class ModelEvaluator:
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or './data/evaluation')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.evaluation_results: Dict[str, Dict] = {}
        
        logger.info(f"Evaluator initialized: {self.output_dir}")
    
    def evaluate_model(
        self,
        model_name: str,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_anomaly_detector: bool = False
    ) -> Dict:
        logger.info(f"📊 Evaluation: {model_name}")
        
        if is_anomaly_detector:
            y_pred_raw = model.predict(X_test)
            y_pred = np.where(y_pred_raw == -1, 1, 0)
            
            if hasattr(model, 'score_samples'):
                y_scores = -model.score_samples(X_test)
            elif hasattr(model, 'decision_function'):
                y_scores = -model.decision_function(X_test)
            else:
                y_scores = None
        else:
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_scores = model.predict_proba(X_test)[:, 1]
            else:
                y_scores = None
        
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        results['true_positives'] = int(tp)
        results['true_negatives'] = int(tn)
        results['false_positives'] = int(fp)
        results['false_negatives'] = int(fn)
        results['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0
        results['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0
        results['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0
        
        if y_scores is not None:
            try:
                fpr, tpr, thresholds = roc_curve(y_test, y_scores)
                results['auc'] = float(auc(fpr, tpr))
                results['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
                
                precision, recall, pr_thresholds = precision_recall_curve(y_test, y_scores)
                results['average_precision'] = float(average_precision_score(y_test, y_scores))
                results['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist()
                }
            except Exception as e:
                logger.warning(f"AUC Error: {e}")
                results['auc'] = 0.0
        
        self.evaluation_results[model_name] = results
        
        return results
    
    def evaluate_all_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        anomaly_models: List[str] = None
    ) -> Dict[str, Dict]:
        if anomaly_models is None:
            anomaly_models = ['isolation_forest', 'one_class_svm', 'lof']
        
        for name, model in models.items():
            if model is not None:
                is_anomaly = name in anomaly_models
                self.evaluate_model(name, model, X_test, y_test, is_anomaly_detector=is_anomaly)
        
        return self.evaluation_results
    
    def plot_confusion_matrix(
        self,
        model_name: str,
        save: bool = True
    ) -> Optional[plt.Figure]:
        if model_name not in self.evaluation_results:
            logger.warning(f"Model not found: {model_name}")
            return None
        
        cm = np.array(self.evaluation_results[model_name]['confusion_matrix'])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign (0)', 'Malicious (1)'],
            yticklabels=['Benign (0)', 'Malicious (1)'],
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / f'confusion_matrix_{model_name}.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_roc_curves(self, save: bool = True) -> Optional[plt.Figure]:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.evaluation_results)))
        
        for (name, results), color in zip(self.evaluation_results.items(), colors):
            if 'roc_curve' in results:
                fpr = results['roc_curve']['fpr']
                tpr = results['roc_curve']['tpr']
                auc_score = results.get('auc', 0)
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{name} (AUC = {auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'roc_curves.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_metrics_comparison(self, save: bool = True) -> Optional[plt.Figure]:
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(self.evaluation_results.keys())
        
        data = []
        for model in models:
            for metric in metrics:
                value = self.evaluation_results[model].get(metric, 0)
                data.append({
                    'Model': model,
                    'Metric': metric.capitalize(),
                    'Value': value
                })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = df[df['Metric'] == metric.capitalize()]['Value'].values
            ax.bar(x + i * width, values, width, label=metric.capitalize())
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metrics Comparison', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'metrics_comparison.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")
        
        return fig
    
    def plot_false_rates(self, save: bool = True) -> Optional[plt.Figure]:
        models = list(self.evaluation_results.keys())
        fpr = [self.evaluation_results[m].get('false_positive_rate', 0) for m in models]
        fnr = [self.evaluation_results[m].get('false_negative_rate', 0) for m in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, fpr, width, label='False Positive Rate', color='coral')
        bars2 = ax.bar(x + width/2, fnr, width, label='False Negative Rate', color='steelblue')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Rate', fontsize=12)
        ax.set_title('Error Rates Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars1, fpr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        for bar, val in zip(bars2, fnr):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save:
            filepath = self.output_dir / 'error_rates.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {filepath}")
        
        return fig
    
    def generate_report(self, save: bool = True) -> str:
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("📊 Model Evaluation Report")
        report_lines.append("=" * 70)
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        for model_name, results in self.evaluation_results.items():
            report_lines.append("-" * 70)
            report_lines.append(f"🔹 Model: {model_name.upper()}")
            report_lines.append("-" * 70)
            report_lines.append(f"   Accuracy: {results['accuracy']:.4f}")
            report_lines.append(f"   Precision: {results['precision']:.4f}")
            report_lines.append(f"   Recall: {results['recall']:.4f}")
            report_lines.append(f"   F1 Score: {results['f1']:.4f}")
            if 'auc' in results:
                report_lines.append(f"   AUC: {results['auc']:.4f}")
            report_lines.append(f"   False Positive Rate: {results['false_positive_rate']:.4f}")
            report_lines.append(f"   False Negative Rate: {results['false_negative_rate']:.4f}")
            report_lines.append("")
            report_lines.append("   Confusion Matrix:")
            cm = np.array(results['confusion_matrix'])
            report_lines.append(f"   TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
            report_lines.append(f"   FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
            report_lines.append("")
        
        report_lines.append("=" * 70)
        report_lines.append("🏆 Conclusion")
        report_lines.append("=" * 70)
        
        if self.evaluation_results:
            best_f1 = max(self.evaluation_results.items(), key=lambda x: x[1]['f1'])
            best_accuracy = max(self.evaluation_results.items(), key=lambda x: x[1]['accuracy'])
            lowest_fpr = min(self.evaluation_results.items(), key=lambda x: x[1]['false_positive_rate'])
            
            report_lines.append(f"   Best F1: {best_f1[0]} ({best_f1[1]['f1']:.4f})")
            report_lines.append(f"   Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
            report_lines.append(f"   Lowest FPR: {lowest_fpr[0]} ({lowest_fpr[1]['false_positive_rate']:.4f})")
        report_lines.append("=" * 70)
        
        report = "\n".join(report_lines)
        
        if save:
            filepath = self.output_dir / 'evaluation_report.txt'
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved: {filepath}")
        
        return report
    
    def save_results(self, filename: str = "evaluation_results.json"):
        filepath = self.output_dir / filename
        
        serializable_results = {}
        for model_name, results in self.evaluation_results.items():
            serializable_results[model_name] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[model_name][key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_results[model_name][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_results[model_name][key] = int(value)
                else:
                    serializable_results[model_name][key] = value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved: {filepath}")
    
    def generate_all_visualizations(self):
        logger.info("📈 Generating visualizations...")
        
        for model_name in self.evaluation_results.keys():
            self.plot_confusion_matrix(model_name)
        
        self.plot_roc_curves()
        self.plot_metrics_comparison()
        self.plot_false_rates()
        
        plt.close('all')
        
        logger.info("✅ All visualizations generated")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Model Evaluation"
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='./data/models',
        help='Models directory'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Test data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./data/evaluation',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    exclude_cols = ['label', 'label_numeric', 'window_start', 'window_end', 'event_count']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_test = df[feature_cols].values
    y_test = df['label_numeric'].values
    
    models_dir = Path(args.models_dir)
    scaler_files = list(models_dir.glob('scaler*.joblib'))
    if scaler_files:
        scaler = joblib.load(scaler_files[-1])
        X_test = scaler.transform(X_test)
    
    models = {}
    model_types = ['isolation_forest', 'one_class_svm', 'lof', 'random_forest', 'xgboost']
    
    for model_type in model_types:
        model_files = list(models_dir.glob(f'{model_type}*.joblib'))
        if model_files:
            models[model_type] = joblib.load(model_files[-1])
    
    evaluator = ModelEvaluator(output_dir=args.output)
    evaluator.evaluate_all_models(models, X_test, y_test)
    
    report = evaluator.generate_report()
    print(report)
    
    evaluator.generate_all_visualizations()
    evaluator.save_results()
    
    logger.info("✅ Finished")


if __name__ == "__main__":
    main()
