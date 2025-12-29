
import os
import sys
import time
import json
import threading
import tracemalloc
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from datetime import datetime
from collections import deque
from dataclasses import dataclass
import logging
import joblib
import numpy as np

try:
    from ..collector.behavior_collector import BehaviorCollector
    from ..features.feature_engineering import FeatureExtractor
except ImportError:
    BehaviorCollector = None
    FeatureExtractor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    timestamp: float
    timestamp_iso: str
    prediction: str
    confidence: float
    model_name: str
    features: Dict[str, float]
    alert_level: str
    latency_ms: float
    memory_mb: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'timestamp_iso': self.timestamp_iso,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'model_name': self.model_name,
            'features': self.features,
            'alert_level': self.alert_level,
            'latency_ms': self.latency_ms,
            'memory_mb': self.memory_mb
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RealtimeDetector:
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest',
        config_path: Optional[str] = None,
        alert_threshold: float = 0.7,
        window_size: float = 10.0,
        max_latency: float = 2.0,
        max_memory_mb: float = 80.0
    ):
        self.model_name = model_name
        self.alert_threshold = alert_threshold
        self.window_size = window_size
        self.max_latency = max_latency
        self.max_memory_mb = max_memory_mb
        
        self.model = None
        self.scaler = None
        self.is_anomaly_detector = model_name in ['isolation_forest', 'one_class_svm', 'lof']
        
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded: {model_path}")
        
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded: {scaler_path}")
        
        self.feature_extractor = FeatureExtractor(window_size=window_size) if FeatureExtractor else None
        
        self._event_buffer: deque = deque(maxlen=10000)
        
        self._detection_history: deque = deque(maxlen=1000)
        
        self._running = False
        self._detection_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self._alert_callback: Optional[Callable[[DetectionResult], None]] = None
        
        self._stats = {
            'total_detections': 0,
            'malicious_count': 0,
            'benign_count': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'current_memory_mb': 0.0
        }
        
        logger.info(f"Real-time Detector initialized")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Window: {window_size}s")
        logger.info(f"   Max Latency: {max_latency}s")
        logger.info(f"   Max RAM: {max_memory_mb} MB")
    
    def set_alert_callback(self, callback: Callable[[DetectionResult], None]):
        self._alert_callback = callback
    
    def add_event(self, event: Dict):
        with self._lock:
            self._event_buffer.append(event)
        
        if self.feature_extractor:
            self.feature_extractor.add_event(event)
    
    def predict(self, features: Dict[str, float]) -> DetectionResult:
        start_time = time.time()
        tracemalloc.start()
        
        try:
            feature_names = list(features.keys())
            feature_values = np.array([[features.get(name, 0) for name in feature_names]])
            
            if self.scaler is not None:
                try:
                    feature_values = self.scaler.transform(feature_values)
                except:
                    pass
            
            prediction = 'benign'
            confidence = 0.5
            
            if self.model is not None:
                if self.is_anomaly_detector:
                    raw_pred = self.model.predict(feature_values)[0]
                    prediction = 'malicious' if raw_pred == -1 else 'benign'
                    
                    if hasattr(self.model, 'score_samples'):
                        score = -self.model.score_samples(feature_values)[0]
                        confidence = min(max(score, 0), 1)
                    elif hasattr(self.model, 'decision_function'):
                        score = -self.model.decision_function(feature_values)[0]
                        confidence = 1 / (1 + np.exp(-score))
                    else:
                        confidence = 0.8 if prediction == 'malicious' else 0.2
                else:
                    prediction = 'malicious' if self.model.predict(feature_values)[0] == 1 else 'benign'
                    if hasattr(self.model, 'predict_proba'):
                        confidence = float(self.model.predict_proba(feature_values)[0][1])
                    else:
                        confidence = 0.9 if prediction == 'malicious' else 0.1
            
            if prediction == 'malicious' and confidence >= self.alert_threshold:
                alert_level = 'danger'
            elif prediction == 'malicious' or confidence >= 0.5:
                alert_level = 'warning'
            else:
                alert_level = 'normal'
            
            latency = (time.time() - start_time) * 1000
            current, peak = tracemalloc.get_traced_memory()
            memory_mb = peak / 1024 / 1024
            tracemalloc.stop()
            
            result = DetectionResult(
                timestamp=time.time() * 1000,
                timestamp_iso=datetime.now().isoformat(),
                prediction=prediction,
                confidence=float(confidence),
                model_name=self.model_name,
                features=features,
                alert_level=alert_level,
                latency_ms=latency,
                memory_mb=memory_mb
            )
            
            self._update_stats(result)
            
            with self._lock:
                self._detection_history.append(result)
            
            if alert_level != 'normal' and self._alert_callback:
                self._alert_callback(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            tracemalloc.stop()
            
            return DetectionResult(
                timestamp=time.time() * 1000,
                timestamp_iso=datetime.now().isoformat(),
                prediction='error',
                confidence=0.0,
                model_name=self.model_name,
                features=features,
                alert_level='warning',
                latency_ms=(time.time() - start_time) * 1000,
                memory_mb=0.0
            )
    
    def detect_current(self) -> DetectionResult:
        if self.feature_extractor:
            features = self.feature_extractor.get_current_features()
        else:
            features = {}
        
        return self.predict(features)
    
    def _update_stats(self, result: DetectionResult):
        self._stats['total_detections'] += 1
        
        if result.prediction == 'malicious':
            self._stats['malicious_count'] += 1
        else:
            self._stats['benign_count'] += 1
        
        n = self._stats['total_detections']
        old_avg = self._stats['avg_latency_ms']
        self._stats['avg_latency_ms'] = old_avg + (result.latency_ms - old_avg) / n
        
        self._stats['max_latency_ms'] = max(self._stats['max_latency_ms'], result.latency_ms)
        self._stats['current_memory_mb'] = result.memory_mb
    
    def _detection_loop(self, interval: float = 1.0):
        logger.info("Starting detection loop")
        
        while self._running:
            try:
                result = self.detect_current()
                
                if result.latency_ms > self.max_latency * 1000:
                    logger.warning(f"⚠️ High latency: {result.latency_ms:.1f}ms")
                
                if result.memory_mb > self.max_memory_mb:
                    logger.warning(f"⚠️ High memory usage: {result.memory_mb:.1f}MB")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1)
        
        logger.info("Detection loop stopped")
    
    def start(self, interval: float = 1.0):
        if self._running:
            logger.warning("Detector already running")
            return
        
        self._running = True
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            args=(interval,),
            daemon=True
        )
        self._detection_thread.start()
        
        logger.info("✅ Real-time Detector started")
    
    def stop(self):
        self._running = False
        if self._detection_thread:
            self._detection_thread.join(timeout=2)
        
        logger.info("✅ Detector stopped")
    
    def get_history(self, count: int = 100) -> List[DetectionResult]:
        with self._lock:
            return list(self._detection_history)[-count:]
    
    def get_stats(self) -> Dict:
        return self._stats.copy()
    
    def get_status(self) -> Dict:
        return {
            'running': self._running,
            'model_name': self.model_name,
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'window_size': self.window_size,
            'alert_threshold': self.alert_threshold,
            'events_in_buffer': len(self._event_buffer),
            'detections_count': len(self._detection_history),
            **self._stats
        }
    
    def print_status(self):
        status = self.get_status()
        
        print("\n" + "=" * 60)
        print("🛡️ Detector Status")
        print("=" * 60)
        print(f"   Running: {'✅' if status['running'] else '❌'}")
        print(f"   Model: {status['model_name']}")
        print(f"   Model Loaded: {'✅' if status['model_loaded'] else '❌'}")
        print(f"   Scaler Loaded: {'✅' if status['scaler_loaded'] else '❌'}")
        print(f"   Event Buffer: {status['events_in_buffer']}")
        print(f"   Total Detections: {status['total_detections']}")
        print(f"   Benign: {status['benign_count']}")
        print(f"   Malicious: {status['malicious_count']}")
        print(f"   Avg Latency: {status['avg_latency_ms']:.2f}ms")
        print(f"   Current Memory: {status['current_memory_mb']:.2f}MB")
        print("=" * 60)


class IntegratedDetector:
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest',
        config_path: Optional[str] = None
    ):
        self.detector = RealtimeDetector(
            model_path=model_path,
            scaler_path=scaler_path,
            model_name=model_name,
            config_path=config_path
        )
        
        if BehaviorCollector:
            self.collector = BehaviorCollector(config_path=config_path)
        else:
            self.collector = None
            logger.warning("Collector not available")
        
        self._running = False
        
        logger.info("Integrated Detector initialized")
    
    def start(self, detection_interval: float = 1.0):
        self._running = True
        
        if self.collector:
            self.collector.start()
        
        self.detector.start(interval=detection_interval)
        
        logger.info("✅ Integrated Detector started")
    
    def stop(self):
        self._running = False
        
        if self.collector:
            self.collector.stop()
        
        self.detector.stop()
        
        logger.info("✅ Integrated Detector stopped")
    
    def get_status(self) -> Dict:
        status = {
            'running': self._running,
            'detector': self.detector.get_status(),
            'collector': self.collector.get_stats() if self.collector else {}
        }
        return status


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-time Detector"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to the model file'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default=None,
        help='Path to the scaler file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='isolation_forest',
        choices=['isolation_forest', 'one_class_svm', 'lof', 'random_forest', 'xgboost'],
        help='Name of the model to use'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration in seconds'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Detection interval in seconds'
    )
    
    args = parser.parse_args()
    
    def on_alert(result: DetectionResult):
        level_icons = {'normal': '🟢', 'warning': '🟡', 'danger': '🔴'}
        icon = level_icons.get(result.alert_level, '❓')
        print(f"\n{icon} ALERT: {result.prediction.upper()}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Latency: {result.latency_ms:.1f}ms")
    
    detector = RealtimeDetector(
        model_path=args.model,
        scaler_path=args.scaler,
        model_name=args.model_name
    )
    detector.set_alert_callback(on_alert)
    
    try:
        print("\n" + "=" * 60)
        print("🛡️ Real-time Detector")
        print("=" * 60)
        
        detector.start(interval=args.interval)
        
        print(f"\n⏳ Running for {args.duration}s...")
        print("Press Ctrl+C to stop\n")
        
        for i in range(args.duration):
            time.sleep(1)
            if (i + 1) % 10 == 0:
                detector.print_status()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Stopped by user")
    
    finally:
        detector.stop()
        detector.print_status()


if __name__ == "__main__":
    main()
