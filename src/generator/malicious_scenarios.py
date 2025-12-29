
import os
import time
import random
import string
import tempfile
import threading
import socket
from typing import Callable, Optional, List
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MaliciousScenarios:
    
    def __init__(self, sandbox_dir: Optional[str] = None):
        self.sandbox_dir = Path(sandbox_dir or tempfile.mkdtemp(prefix="malicious_sim_"))
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = False
        self._threads: List[threading.Thread] = []
        
        logger.info(f"Malicious scenarios initialized in: {self.sandbox_dir}")
        logger.warning("⚠️ This is an educational simulation only")
    
    def _random_string(self, length: int = 10) -> str:
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def _fake_encrypt(self, data: bytes) -> bytes:
        key = 0x42
        return bytes([b ^ key for b in data])
    
    def simulate_file_burst(
        self,
        duration: float = 30,
        files_count: int = 1000,
        callback: Optional[Callable] = None
    ):
        logger.info(f"💥 Starting file burst: {files_count} files")
        
        burst_dir = self.sandbox_dir / "burst_files"
        burst_dir.mkdir(exist_ok=True)
        
        end_time = time.time() + duration
        event_count = 0
        files_created = []
        
        target_per_second = files_count / (duration * 0.6)
        interval = 1.0 / target_per_second if target_per_second > 0 else 0.001
        
        while time.time() < end_time * 0.6 + time.time() * 0.4 and self._running and len(files_created) < files_count:
            try:
                batch_size = random.randint(5, 20)
                for _ in range(batch_size):
                    if len(files_created) >= files_count:
                        break
                    
                    filename = f"burst_{self._random_string(8)}.tmp"
                    filepath = burst_dir / filename
                    
                    content = os.urandom(random.randint(100, 1000))
                    filepath.write_bytes(content)
                    files_created.append(filepath)
                    event_count += 1
                
                if callback:
                    callback("file_burst_create", event_count)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error creating files: {e}")
        
        logger.info(f"🗑️ Starting rapid deletion")
        
        while files_created and self._running and time.time() < end_time:
            try:
                batch_size = random.randint(10, 30)
                for _ in range(min(batch_size, len(files_created))):
                    if not files_created:
                        break
                    
                    filepath = files_created.pop()
                    if filepath.exists():
                        filepath.unlink()
                        event_count += 1
                
                if callback:
                    callback("file_burst_delete", event_count)
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error deleting files: {e}")
        
        logger.info(f"✅ File burst finished: {event_count} events")
        return event_count
    
    def simulate_port_scan(
        self,
        duration: float = 30,
        target_host: str = "127.0.0.1",
        port_range: tuple = (1, 1024),
        callback: Optional[Callable] = None
    ):
        logger.info(f"🔍 Starting port scan: {target_host}")
        
        end_time = time.time() + duration
        event_count = 0
        ports_scanned = []
        
        all_ports = list(range(port_range[0], port_range[1] + 1))
        random.shuffle(all_ports)
        
        while time.time() < end_time and self._running and all_ports:
            try:
                batch_size = random.randint(50, 150)
                
                for _ in range(min(batch_size, len(all_ports))):
                    if not all_ports:
                        break
                    
                    port = all_ports.pop()
                    
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(0.01)
                    
                    try:
                        result = sock.connect_ex((target_host, port))
                        ports_scanned.append((port, result == 0))
                        event_count += 1
                    except:
                        pass
                    finally:
                        sock.close()
                
                if callback:
                    callback("port_scan", event_count)
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error scanning ports: {e}")
        
        open_ports = [p for p, is_open in ports_scanned if is_open]
        logger.info(f"✅ Port scan finished: {event_count} ports, {len(open_ports)} open")
        return event_count
    
    def simulate_sensitive_file_access(
        self,
        duration: float = 30,
        callback: Optional[Callable] = None
    ):
        logger.info("📖 Starting sensitive file access simulation")
        
        sensitive_dir = self.sandbox_dir / "sensitive"
        sensitive_dir.mkdir(exist_ok=True)
        
        fake_sensitive_files = [
            ("fake_passwd", "root:x:0:0:root:/root:/bin/bash\nuser:x:1000:1000:User:/home/user:/bin/bash"),
            ("fake_shadow", "root:$6$fake$hash:18000:0:99999:7:::\nuser:$6$fake$hash:18000:0:99999:7:::"),
            ("fake_ssh_key", "-----BEGIN RSA PRIVATE KEY-----\nFAKE_KEY_DATA_NOT_REAL\n-----END RSA PRIVATE KEY-----"),
            ("fake_credentials", "username=admin\npassword=not_real_password\napi_key=fake_api_key_12345"),
            ("fake_database.db", "FAKE DATABASE CONTENT - NOT REAL DATA"),
        ]
        
        for filename, content in fake_sensitive_files:
            filepath = sensitive_dir / filename
            filepath.write_text(content)
        
        end_time = time.time() + duration
        event_count = 0
        
        while time.time() < end_time and self._running:
            try:
                for filename, _ in fake_sensitive_files:
                    if not self._running:
                        break
                    
                    filepath = sensitive_dir / filename
                    
                    for _ in range(random.randint(5, 20)):
                        _ = filepath.read_text()
                        event_count += 1
                    
                    if callback:
                        callback("sensitive_access", event_count)
                    
                    time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error reading files: {e}")
        
        logger.info(f"✅ Sensitive file access finished: {event_count} reads")
        return event_count
    
    def simulate_ransomware_behavior(
        self,
        duration: float = 30,
        files_to_encrypt: int = 500,
        callback: Optional[Callable] = None
    ):
        logger.info("🔒 Starting ransomware simulation")
        logger.warning("⚠️ Fake encryption only")
        
        victim_dir = self.sandbox_dir / "victim_files"
        victim_dir.mkdir(exist_ok=True)
        
        extensions = ['.txt', '.doc', '.pdf', '.jpg', '.png', '.xlsx']
        created_files = []
        
        for i in range(files_to_encrypt):
            ext = random.choice(extensions)
            filename = f"document_{self._random_string(6)}{ext}"
            filepath = victim_dir / filename
            content = os.urandom(random.randint(100, 2000))
            filepath.write_bytes(content)
            created_files.append(filepath)
        
        end_time = time.time() + duration
        event_count = 0
        encrypted_count = 0
        
        while created_files and self._running and time.time() < end_time:
            try:
                batch_size = random.randint(10, 30)
                
                for _ in range(min(batch_size, len(created_files))):
                    if not created_files:
                        break
                    
                    filepath = created_files.pop()
                    
                    if filepath.exists():
                        content = filepath.read_bytes()
                        event_count += 1
                        
                        encrypted = self._fake_encrypt(content)
                        
                        encrypted_path = filepath.with_suffix(filepath.suffix + '.encrypted')
                        encrypted_path.write_bytes(encrypted)
                        event_count += 1
                        
                        filepath.unlink()
                        event_count += 1
                        
                        encrypted_count += 1
                
                if callback:
                    callback("ransomware_sim", event_count)
                
                time.sleep(0.02)
                
            except Exception as e:
                logger.error(f"Error in ransomware simulation: {e}")
        
        ransom_note = victim_dir / "README_ENCRYPTED.txt"
        ransom_note.write_text("""
⚠️ THIS IS A SIMULATION - NOT REAL RANSOMWARE ⚠️

This is an educational simulation for behavioral detection training.
Your files were NOT actually encrypted.
""")
        
        logger.info(f"✅ Ransomware simulation finished: {encrypted_count} files, {event_count} events")
        return event_count
    
    def simulate_bruteforce(
        self,
        duration: float = 30,
        callback: Optional[Callable] = None
    ):
        logger.info("🔐 Starting brute-force simulation")
        
        wordlist_dir = self.sandbox_dir / "bruteforce"
        wordlist_dir.mkdir(exist_ok=True)
        
        wordlist = wordlist_dir / "wordlist.txt"
        fake_passwords = [f"password{i:04d}" for i in range(10000)]
        wordlist.write_text('\n'.join(fake_passwords))
        
        target_hash = "fake_hash_5f4dcc3b5aa765d61d8327deb882cf99"
        
        end_time = time.time() + duration
        event_count = 0
        attempts = 0
        
        while time.time() < end_time and self._running:
            try:
                passwords = wordlist.read_text().split('\n')
                event_count += 1
                
                batch = random.sample(passwords, min(100, len(passwords)))
                
                for password in batch:
                    if not self._running:
                        break
                    
                    fake_hash = ''.join([str(ord(c) % 10) for c in password])
                    
                    if fake_hash == target_hash:
                        pass
                    
                    attempts += 1
                    event_count += 1
                
                if callback:
                    callback("bruteforce_sim", event_count)
                
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in brute-force simulation: {e}")
        
        logger.info(f"✅ Brute-force finished: {attempts} attempts, {event_count} events")
        return event_count
    
    def run_all_scenarios(
        self,
        duration_per_scenario: float = 30,
        parallel: bool = False,
        callback: Optional[Callable] = None
    ) -> int:
        self._running = True
        total_events = 0
        
        scenarios = [
            ("file_burst", lambda: self.simulate_file_burst(duration=duration_per_scenario, callback=callback)),
            ("port_scan", lambda: self.simulate_port_scan(duration=duration_per_scenario, callback=callback)),
            ("sensitive_access", lambda: self.simulate_sensitive_file_access(duration=duration_per_scenario, callback=callback)),
            ("ransomware", lambda: self.simulate_ransomware_behavior(duration=duration_per_scenario, callback=callback)),
            ("bruteforce", lambda: self.simulate_bruteforce(duration=duration_per_scenario, callback=callback)),
        ]
        
        logger.info(f"🚀 Running {len(scenarios)} malicious scenarios")
        logger.warning("⚠️ This is an educational simulation only")
        
        if parallel:
            results = {}
            threads = []
            
            for name, func in scenarios:
                def run_scenario(n, f):
                    results[n] = f()
                
                t = threading.Thread(target=run_scenario, args=(name, func))
                threads.append(t)
                t.start()
            
            for t in threads:
                t.join()
            
            total_events = sum(results.values())
        else:
            for name, func in scenarios:
                logger.info(f"▶️ Running: {name}")
                events = func()
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
    print("Test Malicious Scenarios")
    print("⚠️ This is an educational simulation only")
    print("=" * 60)
    
    scenarios = MaliciousScenarios()
    
    def on_event(scenario_name, count):
        print(f"  [{scenario_name}] Events: {count}")
    
    try:
        print("\n💥 File burst...")
        scenarios.simulate_file_burst(duration=5, files_count=100, callback=on_event)
        
        print("\n🔍 Port scan...")
        scenarios.simulate_port_scan(duration=5, callback=on_event)
        
        print("\n📖 Sensitive files...")
        scenarios.simulate_sensitive_file_access(duration=5, callback=on_event)
        
        print("\n🔒 Ransomware...")
        scenarios.simulate_ransomware_behavior(duration=5, files_to_encrypt=50, callback=on_event)
        
    finally:
        scenarios.cleanup()
        print("\n✅ Finished")
