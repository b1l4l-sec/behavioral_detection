"""
Behavioral Detection System - Main Launcher
Script principal pour lancer tous les composants du système
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
VENV_STREAMLIT = PROJECT_ROOT / ".venv" / "Scripts" / "streamlit.exe"

def get_python():
    """Get the Python executable path"""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return sys.executable

def get_streamlit():
    """Get the Streamlit executable path"""
    if VENV_STREAMLIT.exists():
        return str(VENV_STREAMLIT)
    return "streamlit"

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n" + "="*60)
    print("  LAUNCHING STREAMLIT DASHBOARD")
    print("="*60)
    streamlit = get_streamlit()
    app_path = PROJECT_ROOT / "src" / "interface" / "streamlit_app.py"
    subprocess.Popen([streamlit, "run", str(app_path), "--server.port=8501"])
    print("  Dashboard starting at http://localhost:8501")
    return True

def run_detector(model="isolation_forest", duration=0):
    """Launch the real-time detector"""
    print("\n" + "="*60)
    print(f"  LAUNCHING REAL-TIME DETECTOR ({model})")
    print("="*60)
    python = get_python()
    subprocess.Popen([
        python, "-m", "src.detector.realtime_detector",
        "--model-name", model,
        "--duration", str(duration)
    ], cwd=str(PROJECT_ROOT))
    print(f"  Detector started with model: {model}")
    return True

def run_generator(benign=1000, malicious=800, duration=10):
    """Launch the dataset generator"""
    print("\n" + "="*60)
    print("  LAUNCHING DATASET GENERATOR")
    print("="*60)
    python = get_python()
    subprocess.run([
        python, "-m", "src.generator.dataset_generator",
        "--benign", str(benign),
        "--malicious", str(malicious),
        "--duration", str(duration)
    ], cwd=str(PROJECT_ROOT))
    return True

def run_trainer(data_path=None):
    """Launch the model trainer"""
    print("\n" + "="*60)
    print("  LAUNCHING MODEL TRAINER")
    print("="*60)
    python = get_python()
    cmd = [python, "-m", "src.models.train_models"]
    if data_path:
        cmd.extend(["--data", data_path])
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return True

def run_collector(duration=60):
    """Launch the behavior collector"""
    print("\n" + "="*60)
    print("  LAUNCHING BEHAVIOR COLLECTOR")
    print("="*60)
    python = get_python()
    subprocess.Popen([
        python, "-m", "src.collector.behavior_collector",
        "--duration", str(duration)
    ], cwd=str(PROJECT_ROOT))
    print(f"  Collector started for {duration} seconds")
    return True

def run_all():
    """Launch all main components"""
    print("\n" + "="*60)
    print("  BEHAVIORAL DETECTION SYSTEM - FULL LAUNCH")
    print("="*60)
    print("  Starting all components...")
    
    run_dashboard()
    time.sleep(2)
    
    run_detector()
    time.sleep(1)
    
    print("\n" + "="*60)
    print("  ALL COMPONENTS STARTED")
    print("="*60)
    print("  - Dashboard: http://localhost:8501")
    print("  - Detector: Running in background")
    print("\n  Press Ctrl+C to stop all processes")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping all processes...")

def run_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("  RUNNING TESTS")
    print("="*60)
    python = get_python()
    subprocess.run([python, "-m", "pytest", "tests/", "-v"], cwd=str(PROJECT_ROOT))
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Detection System - Main Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py dashboard       # Start Streamlit dashboard
  python run.py detector        # Start real-time detector
  python run.py generator       # Generate training data
  python run.py trainer         # Train ML models
  python run.py collector       # Start behavior collector
  python run.py all             # Start all components
  python run.py tests           # Run all tests
        """
    )
    
    parser.add_argument(
        "command",
        choices=["dashboard", "detector", "generator", "trainer", "collector", "all", "tests"],
        help="Component to launch"
    )
    
    parser.add_argument(
        "--model",
        default="isolation_forest",
        choices=["isolation_forest", "random_forest", "xgboost", "one_class_svm", "lof"],
        help="ML model to use (for detector)"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="Duration in seconds (0 = infinite)"
    )
    
    parser.add_argument(
        "--benign",
        type=int,
        default=1000,
        help="Number of benign samples (for generator)"
    )
    
    parser.add_argument(
        "--malicious",
        type=int,
        default=800,
        help="Number of malicious samples (for generator)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  BEHAVIORAL DETECTION SYSTEM")
    print("  Advanced Threat Classification Platform")
    print("="*60)
    
    if args.command == "dashboard":
        run_dashboard()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopping...")
    
    elif args.command == "detector":
        run_detector(model=args.model, duration=args.duration)
        if args.duration == 0:
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n  Stopping...")
    
    elif args.command == "generator":
        run_generator(benign=args.benign, malicious=args.malicious, duration=args.duration or 10)
    
    elif args.command == "trainer":
        run_trainer()
    
    elif args.command == "collector":
        run_collector(duration=args.duration or 60)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n  Stopping...")
    
    elif args.command == "all":
        run_all()
    
    elif args.command == "tests":
        run_tests()

if __name__ == "__main__":
    main()
