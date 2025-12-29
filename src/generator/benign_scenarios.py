
import os
import time
import random
import string
import tempfile
import threading
from typing import Callable, Optional, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BenignScenarios:
    
    def __init__(self, sandbox_dir: Optional[str] = None):
        self.sandbox_dir = Path(sandbox_dir or tempfile.mkdtemp(prefix="benign_"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._threads: List[threading.Thread] = []
        
        logger.info(f"Benign scenarios initialized in: {self.sandbox_dir}")
    
    def _random_string(self, length: int = 10) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def _random_content(self, size: int = 1000) -> str:
        words = ['lorem', 'ipsum', 'dolor', 'sit', 'amet', 'consectetur', 
                 'adipiscing', 'elit', 'sed', 'do', 'eiusmod', 'tempor',
                 'incididunt', 'ut', 'labore', 'et', 'dolore', 'magna', 'aliqua']
        content = []
        while len(' '.join(content)) < size:
            content.append(random.choice(words))
        return ' '.join(content)[:size]
    
    def simulate_web_browsing(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        logger.info("🌐 Starting web browsing simulation")
        
        cache_dir = self.sandbox_dir / "browser_cache"
        cache_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 2.0, "normal": 0.5, "high": 0.1}
        interval = intervals.get(intensity, 0.5)
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                cache_file = cache_dir / f"cache_{self._random_string(8)}.tmp"
                content = self._random_content(random.randint(100, 5000))
                cache_file.write_text(content)
                event_count += 1
                
                cache_files = list(cache_dir.glob("*.tmp"))
                if cache_files:
                    selected = random.choice(cache_files)
                    _ = selected.read_text()
                    event_count += 1
                
                if random.random() < 0.1 and len(cache_files) > 5:
                    oldest = random.choice(cache_files[:5])
                    if oldest.exists():
                        oldest.unlink()
                        event_count += 1
                
                if callback:
                    callback("web_browsing", event_count)
                
                time.sleep(interval + random.uniform(0, interval))
                
            except Exception as e:
                logger.error(f"Error in web browsing simulation: {e}")
        
        logger.info(f"✅ Web browsing finished: {event_count} events")
        return event_count
    
    def simulate_office_work(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        logger.info("📄 Starting office work simulation")
        
        docs_dir = self.sandbox_dir / "documents"
        docs_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 3.0, "normal": 1.0, "high": 0.3}
        interval = intervals.get(intensity, 1.0)
        
        extensions = ['.txt', '.doc', '.csv', '.json']
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                action = random.choice(['create', 'read', 'modify', 'save'])
                
                if action == 'create':
                    ext = random.choice(extensions)
                    doc_file = docs_dir / f"document_{self._random_string(6)}{ext}"
                    content = self._random_content(random.randint(500, 3000))
                    doc_file.write_text(content)
                    event_count += 1
                
                elif action == 'read':
                    doc_files = list(docs_dir.glob("*.*"))
                    if doc_files:
                        selected = random.choice(doc_files)
                        _ = selected.read_text()
                        event_count += 1
                
                elif action == 'modify':
                    doc_files = list(docs_dir.glob("*.*"))
                    if doc_files:
                        selected = random.choice(doc_files)
                        content = selected.read_text()
                        content += f"\n{self._random_content(100)}"
                        selected.write_text(content)
                        event_count += 2
                
                elif action == 'save':
                    doc_files = list(docs_dir.glob("*.*"))
                    if doc_files:
                        selected = random.choice(doc_files)
                        backup = docs_dir / f"{selected.stem}_backup{selected.suffix}"
                        backup.write_text(selected.read_text())
                        event_count += 2
                
                if callback:
                    callback("office_work", event_count)
                
                time.sleep(interval + random.uniform(0, interval * 0.5))
                
            except Exception as e:
                logger.error(f"Error in office work simulation: {e}")
        
        logger.info(f"✅ Office work finished: {event_count} events")
        return event_count
    
    def simulate_compilation(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        logger.info("🔨 Starting compilation simulation")
        
        build_dir = self.sandbox_dir / "build"
        build_dir.mkdir(exist_ok=True)
        src_dir = self.sandbox_dir / "src"
        src_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 1.0, "normal": 0.2, "high": 0.05}
        interval = intervals.get(intensity, 0.2)
        
        end_time = time.time() + duration
        event_count = 0
        
        for i in range(10):
            src_file = src_dir / f"module_{i}.py"
            code = f'''"""Module {i}"""
def function_{i}():
    return {i}

class Class{i}:
    def __init__(self):
        self.value = {i}
'''
            src_file.write_text(code)
        
        while time.time() < end_time and self._running:
            try:
                src_files = list(src_dir.glob("*.py"))
                if src_files:
                    selected = random.choice(src_files)
                    _ = selected.read_text()
                    event_count += 1
                
                obj_file = build_dir / f"obj_{self._random_string(6)}.o"
                obj_file.write_bytes(os.urandom(random.randint(1000, 10000)))
                event_count += 1
                
                tmp_file = build_dir / f"tmp_{self._random_string(4)}.tmp"
                tmp_file.write_text(self._random_content(500))
                event_count += 1
                
                tmp_files = list(build_dir.glob("*.tmp"))
                if len(tmp_files) > 10:
                    for f in tmp_files[:5]:
                        if f.exists():
                            f.unlink()
                            event_count += 1
                
                if callback:
                    callback("compilation", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in compilation simulation: {e}")
        
        logger.info(f"✅ Compilation finished: {event_count} events")
        return event_count
    
    def simulate_file_copy(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        logger.info("📁 Starting file copy simulation")
        
        source_dir = self.sandbox_dir / "source"
        dest_dir = self.sandbox_dir / "destination"
        source_dir.mkdir(exist_ok=True)
        dest_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 2.0, "normal": 0.5, "high": 0.1}
        interval = intervals.get(intensity, 0.5)
        
        for i in range(20):
            f = source_dir / f"file_{i}.dat"
            f.write_bytes(os.urandom(random.randint(100, 5000)))
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                action = random.choice(['copy', 'read', 'move_back'])
                
                if action == 'copy':
                    src_files = list(source_dir.glob("*.*"))
                    if src_files:
                        selected = random.choice(src_files)
                        content = selected.read_bytes()
                        dest_file = dest_dir / f"{selected.stem}_copy_{self._random_string(4)}{selected.suffix}"
                        dest_file.write_bytes(content)
                        event_count += 2
                
                elif action == 'read':
                    all_files = list(source_dir.glob("*.*")) + list(dest_dir.glob("*.*"))
                    if all_files:
                        selected = random.choice(all_files)
                        _ = selected.read_bytes()
                        event_count += 1
                
                elif action == 'move_back':
                    dest_files = list(dest_dir.glob("*.*"))
                    if dest_files and len(dest_files) > 5:
                        selected = random.choice(dest_files)
                        if selected.exists():
                            selected.unlink()
                            event_count += 1
                
                if callback:
                    callback("file_copy", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in file copy simulation: {e}")
        
        logger.info(f"✅ File copy finished: {event_count} events")
        return event_count
    
    def simulate_system_update(
        self,
        duration: float = 30,
        intensity: str = "normal",
        callback: Optional[Callable] = None
    ):
        logger.info("🔄 Starting system update simulation")
        
        update_dir = self.sandbox_dir / "updates"
        update_dir.mkdir(exist_ok=True)
        install_dir = self.sandbox_dir / "installed"
        install_dir.mkdir(exist_ok=True)
        
        intervals = {"low": 2.0, "normal": 0.8, "high": 0.2}
        interval = intervals.get(intensity, 0.8)
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                phase = random.choice(['download', 'extract', 'install', 'cleanup'])
                
                if phase == 'download':
                    pkg_file = update_dir / f"package_{self._random_string(6)}.pkg"
                    pkg_file.write_bytes(os.urandom(random.randint(1000, 10000)))
                    event_count += 1
                
                elif phase == 'extract':
                    pkg_files = list(update_dir.glob("*.pkg"))
                    if pkg_files:
                        selected = random.choice(pkg_files)
                        extract_dir = update_dir / f"extract_{selected.stem}"
                        extract_dir.mkdir(exist_ok=True)
                        for i in range(random.randint(3, 8)):
                            f = extract_dir / f"file_{i}.bin"
                            f.write_bytes(os.urandom(random.randint(100, 1000)))
                            event_count += 1
                
                elif phase == 'install':
                    extract_dirs = [d for d in update_dir.iterdir() if d.is_dir()]
                    if extract_dirs:
                        src = random.choice(extract_dirs)
                        for f in src.glob("*.*"):
                            dest = install_dir / f.name
                            dest.write_bytes(f.read_bytes())
                            event_count += 2
                
                elif phase == 'cleanup':
                    old_pkgs = list(update_dir.glob("*.pkg"))
                    if len(old_pkgs) > 5:
                        for p in old_pkgs[:3]:
                            if p.exists():
                                p.unlink()
                                event_count += 1
                
                if callback:
                    callback("system_update", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in system update simulation: {e}")
        
        logger.info(f"✅ System update finished: {event_count} events")
        return event_count
    
    def run_all_scenarios(
        self,
        duration_per_scenario: float = 30,
        intensity: str = "normal",
        parallel: bool = True,
        callback: Optional[Callable] = None
    ) -> int:
        self._running = True
        total_events = 0
        
        scenarios = [
            ("web_browsing", self.simulate_web_browsing),
            ("office_work", self.simulate_office_work),
            ("compilation", self.simulate_compilation),
            ("file_copy", self.simulate_file_copy),
            ("system_update", self.simulate_system_update)
        ]
        
        logger.info(f"🚀 Running {len(scenarios)} scenarios")
        
        if parallel:
            results = {}
            threads = []
            
            for name, func in scenarios:
                def run_scenario(n, f):
                    results[n] = f(duration=duration_per_scenario, intensity=intensity, callback=callback)
                
                t = threading.Thread(target=run_scenario, args=(name, func))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            total_events = sum(results.values())
        else:
            for name, func in scenarios:
                events = func(duration=duration_per_scenario, intensity=intensity, callback=callback)
                total_events += events
        
        self._running = False
        logger.info(f"✅ All scenarios finished: {total_events} events")
        return total_events
    
    def stop(self):
        self._running = False
    
    def cleanup(self):
        import shutil
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)
            logger.info(f"Cleaned up: {self.sandbox_dir}")


if __name__ == "__main__":
    print("=" * 60)
    print("Test Benign Scenarios")
    print("=" * 60)
    
    scenarios = BenignScenarios()
    
    def on_event(scenario_name, count):
        print(f"  [{scenario_name}] Events: {count}")
    
    try:
        print("\n🌐 Browsing...")
        scenarios.simulate_web_browsing(duration=5, intensity="high", callback=on_event)
        
        print("\n📄 Office...")
        scenarios.simulate_office_work(duration=5, intensity="high", callback=on_event)
        
        print("\n🔨 Compilation...")
        scenarios.simulate_compilation(duration=5, intensity="high", callback=on_event)
        
    finally:
        scenarios.cleanup()
        print("\n✅ Finished")
