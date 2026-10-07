"""Exclude ready-time static objects from cyclic-GC scans during serving.

Automatic GC and reference counting stay enabled. New request objects belong
to ordinary generations; the process-wide permanent generation is released
before Engine teardown. An external owner is never silently unfrozen.
"""
import gc
import threading
import subprocess
import sys

_lock = threading.Lock()
_owned = False
_bootstrap_count = None


def bootstrap_count():
    """CPython3.12 may already keep immortal bootstrap objects permanent."""
    global _bootstrap_count
    if _bootstrap_count is None:
        _bootstrap_count = int(subprocess.check_output([
            sys.executable, '-S', '-c', 'import gc; print(gc.get_freeze_count())'], text=True))
    return _bootstrap_count


class StaticGCGuard:
    def __init__(self):
        self.active = False
        self.frozen_objects = 0

    def start(self):
        global _owned
        with _lock:
            if self.active:
                return self.stats()
            if _owned or gc.get_freeze_count() != bootstrap_count():
                raise RuntimeError('Static GC permanent generation already has an owner')
            if not gc.isenabled():
                raise RuntimeError('Static GC guard requires normal automatic GC')
            gc.collect()
            gc.freeze()
            self.frozen_objects = gc.get_freeze_count()
            self.active = True
            _owned = True
        return self.stats()

    def close(self):
        global _owned
        with _lock:
            if self.active:
                gc.unfreeze()
                # Restore interpreter-owned immortal objects to its permanent
                # generation; model/request objects return to ordinary GC.
                gc.collect()
                self.active = False
                _owned = False

    def stats(self):
        return dict(mode='static_setup_permanent_generation', active=self.active,
                    automatic_gc_enabled=gc.isenabled(), frozen_objects=self.frozen_objects,
                    dynamic_requests_collected=True)
