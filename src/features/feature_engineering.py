
import os
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    
    def __init__(self, window_size: float = 10.0):
        self.window_size = window_size
        self._event_buffer: deque = deque()
        self._last_features: Optional[Dict] = None
        
        logger.info(f"تم تهيئة مستخرج الميزات | Extracteur initialisé: window={window_size}s")
    
    def _calculate_entropy(self, values: List[str]) -> float:
        if not values:
            return 0.0
        
        freq = defaultdict(int)
        for v in values:
            freq[v] += 1
        
        total = len(values)
        entropy = 0.0
        
        for count in freq.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_burstiness(self, timestamps: List[float]) -> float:
        if len(timestamps) < 2:
            return 0.0
        
        sorted_ts = sorted(timestamps)
        intervals = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
        
        if not intervals:
            return 0.0
        
        mean = np.mean(intervals)
        std = np.std(intervals)
        
        if mean == 0:
            return 0.0
        
        burstiness = (std - mean) / (std + mean) if (std + mean) > 0 else 0
        
        return (burstiness + 1) / 2
    
    def extract_features_from_events(self, events: List[Dict]) -> Dict[str, float]:
        if not events:
            return self._get_empty_features()
        
        process_events = [e for e in events if e.get('source') == 'process']
        network_events = [e for e in events if e.get('source') == 'network']
        file_events = [e for e in events if e.get('source') == 'file']
        
        all_timestamps = [e.get('timestamp', 0) for e in events]
        
        if all_timestamps:
            duration = (max(all_timestamps) - min(all_timestamps)) / 1000
            duration = max(duration, 0.001)
        else:
            duration = self.window_size
        
        features = {}
        
        features['file_ops_per_sec'] = len(file_events) / duration
        
        file_paths = [e.get('data', {}).get('path', '') for e in file_events]
        unique_files = len(set(file_paths))
        features['unique_files_ratio'] = unique_files / max(len(file_paths), 1)
        
        create_ops = sum(1 for e in file_events if e.get('data', {}).get('operation') == 'created')
        delete_ops = sum(1 for e in file_events if e.get('data', {}).get('operation') == 'deleted')
        features['delete_create_ratio'] = delete_ops / max(create_ops, 1)
        
        features['path_entropy'] = self._calculate_entropy(file_paths)
        
        extensions = [e.get('data', {}).get('extension', '') for e in file_events]
        features['file_extension_entropy'] = self._calculate_entropy(extensions)
        
        cpu_values = []
        memory_values = []
        io_read_values = []
        io_write_values = []
        
        for e in process_events:
            data = e.get('data', {})
            if 'cpu_percent' in data:
                cpu_values.append(data['cpu_percent'])
            if 'memory_percent' in data:
                memory_values.append(data['memory_percent'])
            if 'io_read_bytes' in data:
                io_read_values.append(data['io_read_bytes'])
            if 'io_write_bytes' in data:
                io_write_values.append(data['io_write_bytes'])
        
        features['cpu_mean'] = np.mean(cpu_values) if cpu_values else 0.0
        
        features['cpu_std'] = np.std(cpu_values) if cpu_values else 0.0
        
        features['memory_mean'] = np.mean(memory_values) if memory_values else 0.0
        
        if io_read_values and len(io_read_values) > 1:
            io_read_rate = (max(io_read_values) - min(io_read_values)) / duration
        else:
            io_read_rate = 0.0
        features['io_read_rate'] = io_read_rate / 1_000_000
        
        if io_write_values and len(io_write_values) > 1:
            io_write_rate = (max(io_write_values) - min(io_write_values)) / duration
        else:
            io_write_rate = 0.0
        features['io_write_rate'] = io_write_rate / 1_000_000
        
        total_io = features['io_read_rate'] + features['io_write_rate']
        if total_io > 0:
            features['io_asymmetry'] = abs(features['io_read_rate'] - features['io_write_rate']) / total_io
        else:
            features['io_asymmetry'] = 0.0
        
        features['net_connections_rate'] = len(network_events) / duration
        
        remote_ports = []
        for e in network_events:
            data = e.get('data', {})
            if 'unique_remote_ports' in data:
                remote_ports.append(data['unique_remote_ports'])
        features['unique_ports_ratio'] = np.mean(remote_ports) if remote_ports else 0.0
        
        net_timestamps = [e.get('timestamp', 0) for e in network_events]
        features['connection_burst'] = self._calculate_burstiness(net_timestamps)
        
        features['burstiness'] = self._calculate_burstiness(all_timestamps)
        
        if len(all_timestamps) > 2:
            sorted_ts = sorted(all_timestamps)
            intervals = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
            if intervals:
                cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                features['temporal_regularity'] = 1 / (1 + cv)
            else:
                features['temporal_regularity'] = 0.5
        else:
            features['temporal_regularity'] = 0.5
        
        features['event_density'] = len(events) / duration
        
        sources = [e.get('source', '') for e in events]
        features['source_diversity'] = len(set(sources)) / 3
        
        self._last_features = features
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        return {
            'file_ops_per_sec': 0.0,
            'unique_files_ratio': 0.0,
            'delete_create_ratio': 0.0,
            'path_entropy': 0.0,
            'file_extension_entropy': 0.0,
            'cpu_mean': 0.0,
            'cpu_std': 0.0,
            'memory_mean': 0.0,
            'io_read_rate': 0.0,
            'io_write_rate': 0.0,
            'io_asymmetry': 0.0,
            'net_connections_rate': 0.0,
            'unique_ports_ratio': 0.0,
            'connection_burst': 0.0,
            'burstiness': 0.0,
            'temporal_regularity': 0.5,
            'event_density': 0.0,
            'source_diversity': 0.0
        }
    
    def get_feature_names(self) -> List[str]:
        return list(self._get_empty_features().keys())
    
    def add_event(self, event: Dict):
        self._event_buffer.append(event)
        
        current_time = event.get('timestamp', 0)
        cutoff = current_time - (self.window_size * 1000)
        
        while self._event_buffer and self._event_buffer[0].get('timestamp', 0) < cutoff:
            self._event_buffer.popleft()
    
    def get_current_features(self) -> Dict[str, float]:
        return self.extract_features_from_events(list(self._event_buffer))
    
    def clear_buffer(self):
        self._event_buffer.clear()


class DatasetFeatureProcessor:
    
    def __init__(self, window_size: float = 10.0, step_size: float = 1.0):
        self.window_size = window_size
        self.step_size = step_size
        self.extractor = FeatureExtractor(window_size=window_size)
        
        logger.info(f"تم تهيئة معالج البيانات | Processeur initialisé: window={window_size}s, step={step_size}s")
    
    def process_jsonl_file(
        self,
        input_file: str,
        output_file: Optional[str] = None
    ) -> pd.DataFrame:
        logger.info(f"معالجة | Traitement: {input_file}")
        
        events = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"تم قراءة {len(events)} حدث | {len(events)} événements lus")
        
        if not events:
            return pd.DataFrame()
        
        events.sort(key=lambda x: x.get('timestamp', 0))
        
        features_list = []
        labels = []
        
        min_ts = events[0].get('timestamp', 0)
        max_ts = events[-1].get('timestamp', 0)
        
        window_ms = self.window_size * 1000
        step_ms = self.step_size * 1000
        
        current_start = min_ts
        
        while current_start + window_ms <= max_ts:
            window_events = [
                e for e in events
                if current_start <= e.get('timestamp', 0) < current_start + window_ms
            ]
            
            if window_events:
                features = self.extractor.extract_features_from_events(window_events)
                features['window_start'] = current_start
                features['window_end'] = current_start + window_ms
                features['event_count'] = len(window_events)
                features_list.append(features)
                
                window_labels = [e.get('label', 'benign') for e in window_events]
                malicious_count = sum(1 for l in window_labels if l == 'malicious')
                label = 'malicious' if malicious_count > len(window_labels) / 2 else 'benign'
                labels.append(label)
            
            current_start += step_ms
        
        df = pd.DataFrame(features_list)
        df['label'] = labels
        df['label_numeric'] = df['label'].map({'benign': 0, 'malicious': 1})
        
        logger.info(f"تم إنشاء {len(df)} نافذة | {len(df)} fenêtres créées")
        
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"تم الحفظ في | Sauvegardé: {output_file}")
        
        return df
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        feature_cols = self.extractor.get_feature_names()
        
        stats = {
            'total_samples': len(df),
            'benign_samples': len(df[df['label'] == 'benign']),
            'malicious_samples': len(df[df['label'] == 'malicious']),
            'features': {}
        }
        
        for col in feature_cols:
            if col in df.columns:
                stats['features'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return stats
    
    def print_statistics(self, df: pd.DataFrame):
        stats = self.get_feature_statistics(df)
        
        print("\n" + "=" * 70)
        print("📊 إحصائيات الميزات | Statistiques des Features")
        print("=" * 70)
        print(f"📦 إجمالي العينات | Total échantillons: {stats['total_samples']}")
        print(f"🌿 حميدة | Bénins: {stats['benign_samples']}")
        print(f"🔴 مشبوهة | Malveillants: {stats['malicious_samples']}")
        print("\n📈 الميزات | Features:")
        print("-" * 70)
        print(f"{'Feature':<30} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)
        
        for name, values in stats['features'].items():
            print(f"{name:<30} {values['mean']:>10.3f} {values['std']:>10.3f} "
                  f"{values['min']:>10.3f} {values['max']:>10.3f}")
        
        print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("اختبار هندسة الميزات | Test Feature Engineering")
    print("=" * 60)
    
    import time
    
    extractor = FeatureExtractor(window_size=10)
    
    test_events = []
    base_time = time.time() * 1000
    
    for i in range(50):
        test_events.append({
            'timestamp': base_time + i * 100,
            'source': 'file',
            'data': {
                'operation': 'created' if i % 3 != 0 else 'deleted',
                'path': f'/test/file_{i % 10}.txt',
                'extension': '.txt'
            },
            'label': 'benign'
        })
    
    for i in range(30):
        test_events.append({
            'timestamp': base_time + i * 150,
            'source': 'process',
            'data': {
                'cpu_percent': 10 + i % 20,
                'memory_percent': 30 + i % 10,
                'io_read_bytes': 1000000 + i * 10000,
                'io_write_bytes': 500000 + i * 5000
            },
            'label': 'benign'
        })
    
    for i in range(20):
        test_events.append({
            'timestamp': base_time + i * 200,
            'source': 'network',
            'data': {
                'unique_remote_ports': i % 10
            },
            'label': 'benign'
        })
    
    features = extractor.extract_features_from_events(test_events)
    
    print("\n📊 الميزات المستخرجة | Features Extraites:")
    print("-" * 40)
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\n✅ تم استخراج {len(features)} ميزة | {len(features)} features extraites")
