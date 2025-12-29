"""
مراقب العمليات | Moniteur de Processus | Process Monitor
يجمع معلومات عن العمليات: CPU, RAM, I/O, threads
Collecte les informations sur les processus: CPU, RAM, I/O, threads
"""

import psutil
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessEvent:
    """
    حدث العملية | Événement de processus
    يمثل لقطة من حالة عملية معينة
    """
    timestamp: float
    timestamp_iso: str
    event_type: str
    pid: int
    name: str
    cpu_percent: float
    memory_percent: float
    memory_rss: int
    io_read_bytes: int
    io_write_bytes: int
    io_read_count: int
    io_write_count: int
    num_threads: int
    num_fds: int
    open_files_count: int
    connections_count: int
    status: str
    
    def to_dict(self) -> Dict:
        """تحويل إلى قاموس | Convertir en dictionnaire"""
        return asdict(self)


class ProcessMonitor:
    """
    مراقب العمليات | Moniteur de Processus
    يراقب العمليات النشطة ويجمع معلوماتها
    Surveille les processus actifs et collecte leurs informations
    """
    
    def __init__(
        self,
        interval: float = 0.5,
        excluded_processes: Optional[List[str]] = None,
        callback: Optional[Callable[[ProcessEvent], None]] = None
    ):
        """
        تهيئة المراقب | Initialisation du moniteur
        
        Args:
            interval: فترة الجمع بالثواني | Intervalle de collecte en secondes
            excluded_processes: العمليات المستثناة | Processus exclus
            callback: دالة الاستدعاء للأحداث | Callback pour les événements
        """
        self.interval = interval
        self.excluded_processes = excluded_processes or [
            "System", "System Idle Process", "Registry", "Idle"
        ]
        self.callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._events: List[ProcessEvent] = []
        self._lock = threading.Lock()
        
        self._prev_io: Dict[int, tuple] = {}
        
        logger.info("تم تهيئة مراقب العمليات | Moniteur de processus initialisé")
    
    def _get_process_info(self, proc: psutil.Process) -> Optional[ProcessEvent]:
        """
        جمع معلومات عملية واحدة | Collecter les infos d'un processus
        
        Args:
            proc: كائن العملية | Objet processus
            
        Returns:
            حدث العملية أو None | Événement ou None
        """
        try:
            name = proc.name()
            if name in self.excluded_processes:
                return None
            
            with proc.oneshot():
                pid = proc.pid
                cpu = proc.cpu_percent()
                mem = proc.memory_percent()
                mem_info = proc.memory_info()
                
                try:
                    io = proc.io_counters()
                    io_read_bytes = io.read_bytes
                    io_write_bytes = io.write_bytes
                    io_read_count = io.read_count
                    io_write_count = io.write_count
                except (psutil.AccessDenied, AttributeError):
                    io_read_bytes = io_write_bytes = 0
                    io_read_count = io_write_count = 0
                
                try:
                    num_threads = proc.num_threads()
                except psutil.AccessDenied:
                    num_threads = 0
                
                try:
                    num_fds = proc.num_fds() if hasattr(proc, 'num_fds') else 0
                except (psutil.AccessDenied, AttributeError):
                    num_fds = 0
                
                try:
                    open_files = len(proc.open_files())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    open_files = 0
                
                try:
                    connections = len(proc.connections())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    connections = 0
                
                try:
                    status = proc.status()
                except psutil.AccessDenied:
                    status = "unknown"
                
                now = datetime.now()
                event = ProcessEvent(
                    timestamp=time.time() * 1000,
                    timestamp_iso=now.isoformat(),
                    event_type="process_snapshot",
                    pid=pid,
                    name=name,
                    cpu_percent=cpu,
                    memory_percent=mem,
                    memory_rss=mem_info.rss,
                    io_read_bytes=io_read_bytes,
                    io_write_bytes=io_write_bytes,
                    io_read_count=io_read_count,
                    io_write_count=io_write_count,
                    num_threads=num_threads,
                    num_fds=num_fds,
                    open_files_count=open_files,
                    connections_count=connections,
                    status=status
                )
                
                return event
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None
    
    def collect_once(self) -> List[ProcessEvent]:
        """
        جمع لقطة واحدة من جميع العمليات
        Collecter un snapshot de tous les processus
        
        Returns:
            قائمة أحداث العمليات | Liste des événements
        """
        events = []
        
        for proc in psutil.process_iter():
            event = self._get_process_info(proc)
            if event:
                events.append(event)
                if self.callback:
                    self.callback(event)
        
        return events
    
    def _monitor_loop(self):
        """
        حلقة المراقبة الرئيسية | Boucle de surveillance principale
        """
        logger.info("بدء حلقة المراقبة | Démarrage de la boucle de surveillance")
        
        while self._running:
            try:
                events = self.collect_once()
                
                with self._lock:
                    self._events.extend(events)
                    if len(self._events) > 10000:
                        self._events = self._events[-10000:]
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"خطأ في حلقة المراقبة | Erreur dans la boucle: {e}")
                time.sleep(1)
        
        logger.info("توقف حلقة المراقبة | Boucle de surveillance arrêtée")
    
    def start(self):
        """
        بدء المراقبة في خيط منفصل
        Démarrer la surveillance dans un thread séparé
        """
        if self._running:
            logger.warning("المراقب يعمل بالفعل | Moniteur déjà en cours")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("تم بدء مراقب العمليات | Moniteur de processus démarré")
    
    def stop(self):
        """
        إيقاف المراقبة | Arrêter la surveillance
        """
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("تم إيقاف مراقب العمليات | Moniteur de processus arrêté")
    
    def get_events(self, clear: bool = False) -> List[ProcessEvent]:
        """
        الحصول على الأحداث المجمعة
        Obtenir les événements collectés
        
        Args:
            clear: مسح الأحداث بعد الحصول عليها | Effacer après récupération
            
        Returns:
            قائمة الأحداث | Liste des événements
        """
        with self._lock:
            events = self._events.copy()
            if clear:
                self._events.clear()
        return events
    
    def get_system_stats(self) -> Dict:
        """
        الحصول على إحصائيات النظام العامة
        Obtenir les statistiques système globales
        
        Returns:
            قاموس الإحصائيات | Dictionnaire des stats
        """
        return {
            "timestamp": time.time() * 1000,
            "timestamp_iso": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": psutil.cpu_count(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage_percent": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
            "process_count": len(psutil.pids()),
            "boot_time": psutil.boot_time()
        }


if __name__ == "__main__":
    print("=" * 60)
    print("اختبار مراقب العمليات | Test du Moniteur de Processus")
    print("=" * 60)
    
    def on_event(event: ProcessEvent):
        if event.cpu_percent > 5:
            print(f"[{event.name}] CPU: {event.cpu_percent:.1f}% | "
                  f"RAM: {event.memory_percent:.1f}% | "
                  f"Threads: {event.num_threads}")
    
    monitor = ProcessMonitor(interval=1.0, callback=on_event)
    
    stats = monitor.get_system_stats()
    print(f"\nإحصائيات النظام | Stats Système:")
    print(f"  CPU: {stats['cpu_percent']}%")
    print(f"  RAM: {stats['memory_percent']}%")
    print(f"  العمليات | Processus: {stats['process_count']}")
    
    print(f"\nالعمليات النشطة | Processus Actifs:")
    events = monitor.collect_once()
    print(f"تم جمع {len(events)} عملية | {len(events)} processus collectés")
    
    print(f"\nبدء المراقبة المستمرة (5 ثوان)...")
    monitor.start()
    time.sleep(5)
    monitor.stop()
    
    events = monitor.get_events()
    print(f"\nإجمالي الأحداث المجمعة: {len(events)}")
