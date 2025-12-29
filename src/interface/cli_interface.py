
import os
import sys
import time
import argparse
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import logging

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("rich library not available")

try:
    from ..detector.realtime_detector import RealtimeDetector, DetectionResult
except ImportError:
    RealtimeDetector = None
    DetectionResult = None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CLIInterface:
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest'
    ):
        if not RICH_AVAILABLE:
            raise ImportError("rich library required")
        
        self.console = Console()
        
        if RealtimeDetector:
            self.detector = RealtimeDetector(
                model_path=model_path,
                scaler_path=scaler_path,
                model_name=model_name
            )
            self.detector.set_alert_callback(self._on_alert)
        else:
            self.detector = None
            self.console.print("[yellow]Detector not available[/]")
        
        self._alerts: List[DetectionResult] = []
        self._max_alerts = 20
        
        self._running = False
    
    def _on_alert(self, result: DetectionResult):
        self._alerts.append(result)
        if len(self._alerts) > self._max_alerts:
            self._alerts.pop(0)
    
    def _get_status_table(self) -> Table:
        table = Table(
            title="System Status",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Indicator", style="cyan", width=30)
        table.add_column("Value", style="green", width=25)
        
        if self.detector:
            status = self.detector.get_status()
            
            running_text = "[green]Running[/]" if status['running'] else "[red]Stopped[/]"
            table.add_row("Status", running_text)
            
            model_status = "[green]Active[/]" if status['model_loaded'] else "[red]Inactive[/]"
            table.add_row("Model", f"{model_status} {status['model_name']}")
            
            table.add_row("Total Detections", str(status['total_detections']))
            table.add_row("Benign", f"[green]{status['benign_count']}[/]")
            table.add_row("Malicious", f"[red]{status['malicious_count']}[/]")
            
            latency = status['avg_latency_ms']
            latency_color = "green" if latency < 100 else "yellow" if latency < 500 else "red"
            table.add_row("Avg Latency", f"[{latency_color}]{latency:.1f}ms[/]")
            
            memory = status['current_memory_mb']
            memory_color = "green" if memory < 40 else "yellow" if memory < 60 else "red"
            table.add_row("Memory", f"[{memory_color}]{memory:.1f}MB[/]")
        
        return table
    
    def _get_alerts_table(self) -> Table:
        table = Table(
            title="Recent Alerts",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold red"
        )
        
        table.add_column("Time", style="dim", width=12)
        table.add_column("Level", width=10)
        table.add_column("Result", width=12)
        table.add_column("Confidence", width=10)
        table.add_column("Latency", width=10)
        
        for alert in reversed(self._alerts[-10:]):
            time_str = datetime.fromtimestamp(alert.timestamp / 1000).strftime("%H:%M:%S")
            
            level_map = {
                'normal': '[green]Normal[/]',
                'warning': '[yellow]Warning[/]',
                'danger': '[red]Danger[/]'
            }
            level = level_map.get(alert.alert_level, 'Unknown')
            
            pred_color = 'red' if alert.prediction == 'malicious' else 'green'
            pred_text = f"[{pred_color}]{alert.prediction}[/]"
            
            conf_color = 'red' if alert.confidence > 0.7 else 'yellow' if alert.confidence > 0.4 else 'green'
            conf_text = f"[{conf_color}]{alert.confidence:.1%}[/]"
            
            table.add_row(
                time_str,
                level,
                pred_text,
                conf_text,
                f"{alert.latency_ms:.1f}ms"
            )
        
        return table
    
    def _get_features_panel(self) -> Panel:
        if self.detector and self.detector.feature_extractor:
            features = self.detector.feature_extractor.get_current_features()
            
            lines = []
            for name, value in list(features.items())[:10]:
                bar_length = int(min(value * 2, 20))
                bar = "█" * bar_length + "░" * (20 - bar_length)
                lines.append(f"{name:<25} {bar} {value:.3f}")
            
            content = "\n".join(lines)
        else:
            content = "No data available"
        
        return Panel(
            content,
            title="Current Features",
            border_style="blue"
        )
    
    def _create_layout(self) -> Layout:
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        
        return layout
    
    def _render(self) -> Layout:
        layout = self._create_layout()
        
        header = Panel(
            Text("Behavioral Detection System", 
                 justify="center", style="bold white on blue"),
            box=box.DOUBLE
        )
        layout["header"].update(header)
        
        layout["left"].update(self._get_status_table())
        
        layout["right"].split_column(
            Layout(self._get_alerts_table(), name="alerts"),
            Layout(self._get_features_panel(), name="features")
        )
        
        footer_text = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Press Ctrl+C to stop"
        footer = Panel(Text(footer_text, justify="center", style="dim"))
        layout["footer"].update(footer)
        
        return layout
    
    def run(self, duration: Optional[int] = None):
        self._running = True
        
        if self.detector:
            self.detector.start(interval=1.0)
        
        start_time = time.time()
        
        try:
            with Live(self._render(), refresh_per_second=2, console=self.console) as live:
                while self._running:
                    live.update(self._render())
                    time.sleep(0.5)
                    
                    if duration and (time.time() - start_time) >= duration:
                        break
        
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopped by user[/]")
        
        finally:
            self._running = False
            if self.detector:
                self.detector.stop()
            
            self._print_summary()
    
    def _print_summary(self):
        self.console.print("\n")
        self.console.print(Panel(
            "[bold]Session Summary[/]",
            box=box.DOUBLE,
            style="cyan"
        ))
        
        if self.detector:
            self.console.print(self._get_status_table())
            
            if self._alerts:
                self.console.print(f"\n[yellow]Total alerts: {len(self._alerts)}[/]")


class SimpleCLI:
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_name: str = 'isolation_forest'
    ):
        if RealtimeDetector:
            self.detector = RealtimeDetector(
                model_path=model_path,
                scaler_path=scaler_path,
                model_name=model_name
            )
            self.detector.set_alert_callback(self._on_alert)
        else:
            self.detector = None
        
        self._alerts = []
        self._running = False
    
    def _on_alert(self, result):
        self._alerts.append(result)
        
        level_icons = {'normal': '🟢', 'warning': '🟡', 'danger': '🔴'}
        icon = level_icons.get(result.alert_level, '?')
        
        print(f"\n{icon} Alert @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"   Result: {result.prediction.upper()}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Latency: {result.latency_ms:.1f}ms")
    
    def _print_status(self):
        if self.detector:
            status = self.detector.get_status()
            
            print("\n" + "=" * 50)
            print("System Status")
            print("=" * 50)
            print(f"   Running: {'Yes' if status['running'] else 'No'}")
            print(f"   Model: {status['model_name']}")
            print(f"   Total Detections: {status['total_detections']}")
            print(f"   Benign: {status['benign_count']}")
            print(f"   Malicious: {status['malicious_count']}")
            print(f"   Avg Latency: {status['avg_latency_ms']:.1f}ms")
            print(f"   Memory: {status['current_memory_mb']:.1f}MB")
            print("=" * 50)
    
    def run(self, duration: Optional[int] = None):
        print("\n" + "=" * 60)
        print("Behavioral Detection System")
        print("=" * 60)
        
        self._running = True
        
        if self.detector:
            self.detector.start(interval=1.0)
        
        start_time = time.time()
        status_interval = 10
        last_status = time.time()
        
        try:
            print("\nDetection running...")
            print("Press Ctrl+C to stop\n")
            
            while self._running:
                time.sleep(1)
                
                if time.time() - last_status >= status_interval:
                    self._print_status()
                    last_status = time.time()
                
                if duration and (time.time() - start_time) >= duration:
                    break
        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        
        finally:
            self._running = False
            if self.detector:
                self.detector.stop()
            
            self._print_status()
            print(f"\nTotal alerts: {len(self._alerts)}")


def main():
    parser = argparse.ArgumentParser(
        description="CLI Interface for Detection"
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model path'
    )
    parser.add_argument(
        '--scaler',
        type=str,
        default=None,
        help='Scaler path'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='isolation_forest',
        help='Model name'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=None,
        help='Duration in seconds'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Use simple interface'
    )
    
    args = parser.parse_args()
    
    if args.simple or not RICH_AVAILABLE:
        cli = SimpleCLI(
            model_path=args.model,
            scaler_path=args.scaler,
            model_name=args.model_name
        )
    else:
        cli = CLIInterface(
            model_path=args.model,
            scaler_path=args.scaler,
            model_name=args.model_name
        )
    
    cli.run(duration=args.duration)


if __name__ == "__main__":
    main()
