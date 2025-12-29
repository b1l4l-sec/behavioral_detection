"""
Dataset Generator
Combines benign and malicious scenarios to generate training data
"""

import os
import sys
import json
import time
import yaml
import argparse
import threading
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging

from .benign_scenarios import BenignScenarios
from .malicious_scenarios import MaliciousScenarios

try:
    from ..collector.behavior_collector import BehaviorCollector
except ImportError:
    BehaviorCollector = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """
    Dataset Generator
    Generates balanced training dataset
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the generator
        
        Args:
            output_dir: Output directory
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        self.output_dir = Path(output_dir or self.config.get('output_dir', './data'))
        self.raw_dir = self.output_dir / 'raw'
        self.processed_dir = self.output_dir / 'processed'
        self.sandbox_dir = self.output_dir / 'sandbox'
        
        for d in [self.raw_dir, self.processed_dir, self.sandbox_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.benign_scenarios = BenignScenarios(
            sandbox_dir=str(self.sandbox_dir / 'benign')
        )
        self.malicious_scenarios = MaliciousScenarios(
            sandbox_dir=str(self.sandbox_dir / 'malicious')
        )
        
        self._stats = {
            'benign_events': 0,
            'malicious_events': 0,
            'generation_time': 0
        }
        
        logger.info(f"Dataset generator initialized: {self.output_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration
        """
        default_config = {
            'output_dir': './data',
            'output_format': 'jsonl',
            'benign_events': 10000,
            'malicious_events': 8000,
            'duration_per_scenario': 30,
            'intensity': 'normal'
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded = yaml.safe_load(f)
                    if loaded and 'dataset' in loaded:
                        default_config.update(loaded['dataset'])
            except Exception as e:
                logger.warning(f"Error loading config: {e}")
        
        return default_config
    
    def generate_benign_dataset(
        self,
        target_events: int = 10000,
        duration_per_scenario: float = 30,
        output_file: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Generate benign dataset
        
        Args:
            target_events: Target number of events
            duration_per_scenario: Duration per scenario
            output_file: Output file
            
        Returns:
            File path and event count
        """
        logger.info("=" * 60)
        logger.info("🌿 Generating benign data")
        logger.info("=" * 60)
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.raw_dir / f"benign_{timestamp}.jsonl")
        
        events = []
        event_count = 0
        
        def on_event(scenario_name, count):
            nonlocal event_count
            event_count = count
        
        scenarios = [
            ("web_browsing", self.benign_scenarios.simulate_web_browsing),
            ("office_work", self.benign_scenarios.simulate_office_work),
            ("compilation", self.benign_scenarios.simulate_compilation),
            ("file_copy", self.benign_scenarios.simulate_file_copy),
            ("system_update", self.benign_scenarios.simulate_system_update),
        ]
        
        total_events = 0
        
        for name, func in scenarios:
            logger.info(f"▶️ Running: {name}")
            
            remaining = target_events - total_events
            if remaining <= 0:
                break
            
            adjusted_duration = min(duration_per_scenario, max(5, remaining / 100))
            
            start_count = event_count
            func(duration=adjusted_duration, intensity='high', callback=on_event)
            scenario_events = event_count - start_count
            
            for i in range(scenario_events):
                event = {
                    'timestamp': time.time() * 1000 + i,
                    'timestamp_iso': datetime.now().isoformat(),
                    'source': 'generated',
                    'event_type': name,
                    'scenario': name,
                    'label': 'benign',
                    'data': {
                        'scenario_name': name,
                        'event_index': i
                    }
                }
                events.append(event)
            
            total_events += scenario_events
            logger.info(f"   ✅ {scenario_events} events")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        
        self._stats['benign_events'] = total_events
        logger.info(f"✅ Saved {total_events} events to: {output_file}")
        
        return output_file, total_events
    
    def generate_malicious_dataset(
        self,
        target_events: int = 8000,
        duration_per_scenario: float = 30,
        output_file: Optional[str] = None
    ) -> Tuple[str, int]:
        """
        Generate malicious dataset
        
        Args:
            target_events: Target number of events
            duration_per_scenario: Duration per scenario
            output_file: Output file
            
        Returns:
            File path and event count
        """
        logger.info("=" * 60)
        logger.info("🔴 Generating malicious data")
        logger.info("⚠️ This is an educational simulation only")
        logger.info("=" * 60)
        
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = str(self.raw_dir / f"malicious_{timestamp}.jsonl")
        
        events = []
        event_count = 0
        
        def on_event(scenario_name, count):
            nonlocal event_count
            event_count = count
        
        scenarios = [
            ("file_burst", lambda: self.malicious_scenarios.simulate_file_burst(
                duration=duration_per_scenario, files_count=500, callback=on_event)),
            ("port_scan", lambda: self.malicious_scenarios.simulate_port_scan(
                duration=duration_per_scenario, callback=on_event)),
            ("sensitive_access", lambda: self.malicious_scenarios.simulate_sensitive_file_access(
                duration=duration_per_scenario, callback=on_event)),
            ("ransomware", lambda: self.malicious_scenarios.simulate_ransomware_behavior(
                duration=duration_per_scenario, files_to_encrypt=200, callback=on_event)),
            ("bruteforce", lambda: self.malicious_scenarios.simulate_bruteforce(
                duration=duration_per_scenario, callback=on_event)),
        ]
        
        total_events = 0
        
        for name, func in scenarios:
            logger.info(f"▶️ Running: {name}")
            
            start_count = event_count
            func()
            scenario_events = event_count - start_count
            
            for i in range(scenario_events):
                event = {
                    'timestamp': time.time() * 1000 + i,
                    'timestamp_iso': datetime.now().isoformat(),
                    'source': 'generated',
                    'event_type': name,
                    'scenario': name,
                    'label': 'malicious',
                    'data': {
                        'scenario_name': name,
                        'event_index': i,
                        'attack_type': name
                    }
                }
                events.append(event)
            
            total_events += scenario_events
            logger.info(f"   ✅ {scenario_events} events")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        
        self._stats['malicious_events'] = total_events
        logger.info(f"✅ Saved {total_events} events to: {output_file}")
        
        return output_file, total_events
    
    def generate_combined_dataset(
        self,
        benign_events: int = 10000,
        malicious_events: int = 8000,
        duration_per_scenario: float = 30,
        shuffle: bool = True
    ) -> Tuple[str, Dict]:
        """
        Generate combined dataset
        
        Args:
            benign_events: Number of benign events
            malicious_events: Number of malicious events
            duration_per_scenario: Duration per scenario
            shuffle: Shuffle data
            
        Returns:
            File path and statistics
        """
        import random
        
        logger.info("=" * 60)
        logger.info("🎯 Generating combined dataset")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        benign_file, actual_benign = self.generate_benign_dataset(
            target_events=benign_events,
            duration_per_scenario=duration_per_scenario
        )
        
        malicious_file, actual_malicious = self.generate_malicious_dataset(
            target_events=malicious_events,
            duration_per_scenario=duration_per_scenario
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_file = str(self.processed_dir / f"combined_dataset_{timestamp}.jsonl")
        
        all_events = []
        
        with open(benign_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_events.append(json.loads(line))
        
        with open(malicious_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_events.append(json.loads(line))
        
        if shuffle:
            random.shuffle(all_events)
        
        with open(combined_file, 'w', encoding='utf-8') as f:
            for event in all_events:
                f.write(json.dumps(event, ensure_ascii=False) + '\n')
        
        generation_time = time.time() - start_time
        self._stats['generation_time'] = generation_time
        
        stats = {
            'benign_events': actual_benign,
            'malicious_events': actual_malicious,
            'total_events': len(all_events),
            'generation_time': generation_time,
            'combined_file': combined_file,
            'benign_file': benign_file,
            'malicious_file': malicious_file
        }
        
        stats_file = str(self.processed_dir / f"dataset_stats_{timestamp}.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self._print_summary(stats)
        
        return combined_file, stats
    
    def _print_summary(self, stats: Dict):
        """
        Print generation summary
        """
        print("\n" + "=" * 60)
        print("📊 Dataset Generation Summary")
        print("=" * 60)
        print(f"🌿 Benign events: {stats['benign_events']}")
        print(f"🔴 Malicious events: {stats['malicious_events']}")
        print(f"📦 Total events: {stats['total_events']}")
        print(f"⏱️  Generation time: {stats['generation_time']:.1f}s")
        print(f"📁 Combined file: {stats['combined_file']}")
        print("=" * 60)
    
    def cleanup(self):
        """
        Clean up simulation files
        """
        self.benign_scenarios.cleanup()
        self.malicious_scenarios.cleanup()
        logger.info("Cleaned up")
    
    def validate_dataset(self, filepath: str) -> Dict:
        """
        Validate dataset
        
        Args:
            filepath: File path
            
        Returns:
            Validation results
        """
        logger.info(f"🔍 Validating: {filepath}")
        
        stats = {
            'total_lines': 0,
            'valid_events': 0,
            'invalid_events': 0,
            'benign_count': 0,
            'malicious_count': 0,
            'scenarios': {},
            'errors': []
        }
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    stats['total_lines'] += 1
                    try:
                        event = json.loads(line)
                        stats['valid_events'] += 1
                        
                        label = event.get('label', 'unknown')
                        if label == 'benign':
                            stats['benign_count'] += 1
                        elif label == 'malicious':
                            stats['malicious_count'] += 1
                        
                        scenario = event.get('scenario', event.get('event_type', 'unknown'))
                        stats['scenarios'][scenario] = stats['scenarios'].get(scenario, 0) + 1
                        
                    except json.JSONDecodeError as e:
                        stats['invalid_events'] += 1
                        stats['errors'].append(f"Line {i}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Validation error: {e}")
            stats['errors'].append(str(e))
        
        print("\n" + "=" * 60)
        print("📋 Validation Results")
        print("=" * 60)
        print(f"📄 Total lines: {stats['total_lines']}")
        print(f"✅ Valid events: {stats['valid_events']}")
        print(f"❌ Invalid events: {stats['invalid_events']}")
        print(f"🌿 Benign: {stats['benign_count']}")
        print(f"🔴 Malicious: {stats['malicious_count']}")
        print("\n📊 Scenarios:")
        for scenario, count in sorted(stats['scenarios'].items()):
            print(f"   - {scenario}: {count}")
        print("=" * 60)
        
        return stats


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(
        description="Dataset Generator"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./data',
        help='Output directory'
    )
    parser.add_argument(
        '--benign', '-b',
        type=int,
        default=10000,
        help='Number of benign events'
    )
    parser.add_argument(
        '--malicious', '-m',
        type=int,
        default=8000,
        help='Number of malicious events'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=30,
        help='Duration per scenario in seconds'
    )
    parser.add_argument(
        '--validate',
        type=str,
        default=None,
        help='Validate a file'
    )
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(
        output_dir=args.output,
        config_path=args.config
    )
    
    try:
        if args.validate:
            generator.validate_dataset(args.validate)
        else:
            combined_file, stats = generator.generate_combined_dataset(
                benign_events=args.benign,
                malicious_events=args.malicious,
                duration_per_scenario=args.duration
            )
            
            generator.validate_dataset(combined_file)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Stopped by user")
    
    finally:
        generator.cleanup()


if __name__ == "__main__":
    main()
