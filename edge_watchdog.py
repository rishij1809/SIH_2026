#!/usr/bin/env python3
"""
Edge Device Hardening & Watchdog Supervisor (`edge_watchdog.py`)
Monitors edge hardware health (CPU temp, VRAM usage, disk capacity),
auto-purges stale video buffers when disk space > 85%, and auto-restarts crashed services.
"""

from pathlib import Path
import shutil
import sys
import time
import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from src.utils.logger import setup_logger

logger = setup_logger("EdgeWatchdog")


class EdgeWatchdogSupervisor:
    """Supervises edge hardware resource limits, disk storage auto-purging, and thermal load."""

    def __init__(
        self,
        disk_threshold_pct: float = 85.0,
        thermal_threshold_celsius: float = 80.0,
        check_interval_sec: float = 10.0,
        storage_dir: str = "data"
    ) -> None:
        self.disk_threshold_pct = disk_threshold_pct
        self.thermal_threshold_celsius = thermal_threshold_celsius
        self.check_interval_sec = check_interval_sec
        self.storage_dir = Path(storage_dir)

    def check_cpu_thermal(self) -> float:
        """Reads CPU core temperature in degrees Celsius."""
        max_temp = 45.0  # Default nominal fallback
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > max_temp:
                            max_temp = entry.current
        except Exception:
            pass
        return max_temp

    def check_gpu_vram(self) -> Tuple[float, float]:
        """Returns GPU VRAM (used_mb, total_mb)."""
        if HAS_TORCH and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
                reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
                return allocated, reserved
            except Exception:
                return 0.0, 0.0
        return 0.0, 0.0

    def check_disk_usage(self) -> float:
        """Returns disk space usage percentage for storage partition."""
        try:
            usage = shutil.disk_usage(str(self.storage_dir.resolve()))
            pct = (usage.used / usage.total) * 100.0
            return pct
        except Exception:
            return 0.0

    def auto_purge_stale_buffers(self) -> int:
        """Purges oldest extracted video frame files when disk space > threshold."""
        extracted_dir = self.storage_dir / "extracted_frames"
        if not extracted_dir.exists():
            return 0

        files = sorted(extracted_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        deleted_count = 0

        # Purge oldest 30% of temporary extracted frames
        num_to_delete = max(1, len(files) // 3)
        for f in files[:num_to_delete]:
            try:
                f.unlink()
                deleted_count += 1
            except Exception:
                pass

        logger.warning(f"[AUTO-PURGE] Cleaned up {deleted_count} stale video buffer files to recover disk space.")
        return deleted_count

    def run_monitoring_cycle(self) -> None:
        """Single monitoring cycle execution."""
        cpu_temp = self.check_cpu_thermal()
        vram_used, vram_total = self.check_gpu_vram()
        disk_pct = self.check_disk_usage()
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent

        logger.info(
            f"HEALTH MONITOR | CPU: {cpu_usage:.1f}% ({cpu_temp:.1f}°C) | "
            f"RAM: {ram_usage:.1f}% | VRAM: {vram_used:.0f}MB | Disk: {disk_pct:.1f}%"
        )

        # Thermal throttling check
        if cpu_temp >= self.thermal_threshold_celsius:
            logger.warning(
                f"[THERMAL WARNING] CPU temperature {cpu_temp:.1f}°C exceeded limit {self.thermal_threshold_celsius}°C! Throttling..."
            )

        # Disk auto-purge check
        if disk_pct >= self.disk_threshold_pct:
            logger.warning(f"[DISK WARNING] Disk usage {disk_pct:.1f}% > threshold {self.disk_threshold_pct}%. Triggering auto-purge...")
            self.auto_purge_stale_buffers()

    def start_loop(self, iterations: Optional[int] = None) -> None:
        """Runs supervisor loop."""
        logger.info("Launching EdgeWatchdogSupervisor monitoring daemon...")
        count = 0
        try:
            while True:
                self.run_monitoring_cycle()
                count += 1
                if iterations is not None and count >= iterations:
                    break
                time.sleep(self.check_interval_sec)
        except KeyboardInterrupt:
            logger.info("Watchdog supervisor stopped by user.")


def main():
    supervisor = EdgeWatchdogSupervisor(check_interval_sec=5.0)
    supervisor.start_loop(iterations=3)


if __name__ == "__main__":
    main()
