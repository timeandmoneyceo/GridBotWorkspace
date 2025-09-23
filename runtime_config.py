"""
Runtime Config Utilities

Lightweight file-backed runtime configuration that can be reloaded by long-running
processes without restart. Default config file: runtime_config.json at project root.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Callable, Dict, Optional

_DEFAULT_PATH = os.path.join(os.getcwd(), "runtime_config.json")
_lock = threading.Lock()
_cache: Dict[str, Any] = {}
_cache_mtime: float = 0.0
_config_path = _DEFAULT_PATH


def set_config_path(path: str) -> None:
    global _config_path
    _config_path = path


def _safe_load(path: Optional[str] = None) -> Dict[str, Any]:
    path = path or _config_path
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_all(path: Optional[str] = None) -> Dict[str, Any]:
    global _cache, _cache_mtime
    path = path or _config_path
    with _lock:
        try:
            mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
            if mtime != _cache_mtime:
                _cache = _safe_load(path)
                _cache_mtime = mtime
        except Exception:
            pass
        return dict(_cache)


def get_param(key: str, default: Any = None, path: Optional[str] = None) -> Any:
    data = get_all(path)
    # parameters stored under 'parameters' key
    params = data.get("parameters", {}) if isinstance(data, dict) else {}
    return params.get(key, default)


def update_parameters(updates: Dict[str, Any], path: Optional[str] = None) -> bool:
    path = path or _config_path
    with _lock:
        data = _safe_load(path)
        params = data.get("parameters", {}) if isinstance(data, dict) else {}
        params.update(updates)
        data["parameters"] = params
        data["last_update"] = time.time()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            # update cache
            global _cache, _cache_mtime
            _cache = data
            _cache_mtime = os.path.getmtime(path)
            return True
        except Exception:
            return False


def watch_and_callback(callback: Callable[[Dict[str, Any]], None], interval_sec: int = 3, path: Optional[str] = None, stop_event: Optional[threading.Event] = None) -> threading.Thread:
    """Start a background watcher that invokes callback when file changes."""
    path = path or _config_path
    ev = stop_event or threading.Event()

    def _loop():
        prev = 0.0
        while not ev.is_set():
            try:
                cur = os.path.getmtime(path) if os.path.exists(path) else 0.0
                if cur != prev:
                    prev = cur
                    cfg = get_all(path)
                    try:
                        callback(cfg)
                    except Exception:
                        pass
            except Exception:
                pass
            ev.wait(interval_sec)

    th = threading.Thread(target=_loop, daemon=True)
    th.start()
    return th
