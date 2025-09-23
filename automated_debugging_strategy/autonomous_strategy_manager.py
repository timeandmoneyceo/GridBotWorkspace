"""
Autonomous Strategy Manager

Coordinates real-time monitoring, adaptive decision-making, continuous learning,
and automated optimization to continuously evolve the GridBot strategy and targeted scripts.

Lightweight, dependency-safe design: integrates with existing components (LLM interface,
SafeFileEditor, EnhancedOptimizationSystem, FileManagementSystem, DebugLogParser) and
persists learning state to disk. Designed to run as a background daemon with safe
start/stop and interval-based ticks.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

# Runtime config (project root)
try:
    import runtime_config
except Exception:  # fallback shim if unavailable at import time
    runtime_config = None  # type: ignore

# Local imports (absolute to support script execution)
# Support both package and script execution import styles
try:
    from .enhanced_optimization_system import EnhancedOptimizationSystem as _EnhancedOptimizationSystem
    from .file_management_system import FileManagementSystem
    from .debug_log_parser import DebugLogParser
except ImportError:  # When executed as scripts (no package context)
    from enhanced_optimization_system import EnhancedOptimizationSystem as _EnhancedOptimizationSystem
    from file_management_system import FileManagementSystem
    from debug_log_parser import DebugLogParser


@dataclass
class StrategyState:
    """Operational state for the strategy and learning engine."""
    current_profile: str = "default"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "grid_size": 10,
        "risk": 0.02,
        "max_positions": 5,
        "enable_dynamic_grid": True,
    })
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    last_actions: List[Dict[str, Any]] = field(default_factory=list)


class RealTimeMonitor:
    """Aggregates bot and (optionally) market metrics.

    This is intentionally resilient: if no live sources are available, it falls back to
    logs and cached reports. You can later plug in exchange/websocket sources.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.logger = logging.getLogger(__name__)
        self.log_parser = DebugLogParser()

    def get_metrics(self) -> Dict[str, Any]:
        """Return a unified set of metrics used by the decision engine.

        Structure:
        {
          'timestamp': epoch,
          'pnl': float,
          'win_rate': float,
          'drawdown': float,
          'volatility': float,
          'error_rate': float,
          'latency_ms': float,
          'signals_per_min': float
        }
        """
        ts = time.time()

        if http_url := os.environ.get("GRIDBOT_METRICS_HTTP"):
            try:
                import requests  # optional dependency; used only if available
                r = requests.get(http_url, timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    return {
                        "timestamp": ts,
                        "pnl": float(data.get("pnl", 0.0)),
                        "win_rate": float(data.get("win_rate", 0.0)),
                        "drawdown": float(data.get("drawdown", 0.0)),
                        "volatility": float(data.get("volatility", 0.0)),
                        "error_rate": float(data.get("error_rate", 0.0)),
                        "latency_ms": float(data.get("latency_ms", 0.0)),
                        "signals_per_min": float(data.get("signals_per_min", 0.0)),
                    }
            except Exception as e:
                self.logger.debug(f"[MONITOR] HTTP metrics failed: {e}")

        # Try performance_metrics.json if present
        perf_file = os.path.join(self.base_dir, "reports", "performance_metrics.json")
        if os.path.exists(perf_file):
            try:
                with open(perf_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                # Ensure defaults
                return {
                    "timestamp": ts,
                    "pnl": float(data.get("pnl", 0.0)),
                    "win_rate": float(data.get("win_rate", 0.0)),
                    "drawdown": float(data.get("drawdown", 0.0)),
                    "volatility": float(data.get("volatility", 0.0)),
                    "error_rate": float(data.get("error_rate", 0.0)),
                    "latency_ms": float(data.get("latency_ms", 0.0)),
                    "signals_per_min": float(data.get("signals_per_min", 0.0)),
                }
            except Exception as e:
                self.logger.warning(f"[MONITOR] Failed to load performance_metrics.json: {e}")

        # Fallback: attempt to scan for latest debug report json to infer error rate
        error_rate = 0.0
        latency_ms = 0.0
        try:
            candidate_dirs = [self.base_dir, os.path.join(self.base_dir, "automated_debugging_strategy")]
            latest_file = None
            latest_mtime = 0.0
            for d in candidate_dirs:
                if not os.path.isdir(d):
                    continue
                for name in os.listdir(d):
                    if name.lower().startswith("debug_report") and name.lower().endswith(".json"):
                        fp = os.path.join(d, name)
                        try:
                            mt = os.path.getmtime(fp)
                            if mt > latest_mtime:
                                latest_mtime = mt
                                latest_file = fp
                        except Exception:
                            continue
            if latest_file:
                with open(latest_file, "r", encoding="utf-8") as f:
                    rep = json.load(f)
                errs = rep.get("errors", []) if isinstance(rep, dict) else []
                error_rate = min(1.0, float(len(errs)) / 100.0)
        except Exception as e:
            self.logger.debug(f"[MONITOR] Debug report scan failed: {e}")

        # Conservative synthetic defaults; replace when live feeds exist
        return {
            "timestamp": ts,
            "pnl": 0.0,
            "win_rate": 0.0,
            "drawdown": 0.0,
            "volatility": 0.0,
            "error_rate": error_rate,
            "latency_ms": latency_ms,
            "signals_per_min": 0.0,
        }


class AdaptiveDecisionEngine:
    """Converts metrics into actionable strategy adjustments."""

    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.logger = logging.getLogger(__name__)
        self.thresholds = thresholds or {
            "max_drawdown": 0.15,
            "min_win_rate": 0.45,
            "max_error_rate": 0.1,
            "max_latency_ms": 500.0,
        }

    def decide(self, metrics: Dict[str, Any], state: StrategyState) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        # Risk control based on drawdown
        if float(metrics.get("drawdown", 0.0)) > self.thresholds["max_drawdown"]:
            actions.append({"type": "adjust_param", "key": "risk", "delta": -0.005, "min": 0.005})

        # Grid adaptation based on volatility
        vol = float(metrics.get("volatility", 0.0))
        if vol >= 0.05:
            actions.append({"type": "adjust_param", "key": "grid_size", "delta": +2, "max": 50})
        elif vol <= 0.02:
            actions.append({"type": "adjust_param", "key": "grid_size", "delta": -1, "min": 3})

        # Error mitigation
        if float(metrics.get("error_rate", 0.0)) > self.thresholds["max_error_rate"]:
            actions.append({"type": "trigger_optimization", "reason": "high_error_rate"})

        # Latency guardrails
        if float(metrics.get("latency_ms", 0.0)) > self.thresholds["max_latency_ms"]:
            actions.append({"type": "trigger_optimization", "reason": "high_latency"})

        # Performance improvement
        if float(metrics.get("win_rate", 0.0)) < self.thresholds["min_win_rate"]:
            actions.append({"type": "explore_strategy_variant", "variant": "conservative"})

        self.logger.debug(f"[DECIDE] actions={actions}")
        return actions


class ContinuousLearningEngine:
    """Tracks outcomes and updates strategy parameter weights over time."""

    def __init__(self, state_path: str):
        self.logger = logging.getLogger(__name__)
        self.state_path = state_path

    def load(self) -> StrategyState:
        try:
            if os.path.exists(self.state_path):
                with open(self.state_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return StrategyState(
                    current_profile=raw.get("current_profile", "default"),
                    parameters=raw.get("parameters", {}),
                    performance_history=raw.get("performance_history", []),
                    last_actions=raw.get("last_actions", []),
                )
        except Exception as e:
            self.logger.warning(f"[LEARN] Failed to load state: {e}")
        return StrategyState()

    def save(self, state: StrategyState) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump({
                    "current_profile": state.current_profile,
                    "parameters": state.parameters,
                    "performance_history": state.performance_history[-1000:],
                    "last_actions": state.last_actions[-200:],
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"[LEARN] Failed to save state: {e}")

    def update_from_outcome(self, state: StrategyState, metrics: Dict[str, Any], actions: List[Dict[str, Any]]):
        record = {
            "ts": metrics.get("timestamp", time.time()),
            "metrics": metrics,
            "actions": actions,
        }
        state.performance_history.append(record)
        state.last_actions = actions
        # Simple learning heuristic: adjust risk slightly towards profit, away from drawdowns
        pnl = float(metrics.get("pnl", 0.0))
        dd = float(metrics.get("drawdown", 0.0))
        if pnl > 0 and dd < 0.05:
            state.parameters["risk"] = max(0.005, min(0.05, state.parameters.get("risk", 0.02) + 0.001))
        elif dd > 0.1:
            state.parameters["risk"] = max(0.005, state.parameters.get("risk", 0.02) - 0.001)


class AutoOptimizationPipeline:
    """Wrapper around EnhancedOptimizationSystem to propose/apply improvements."""

    def __init__(self, base_dir: str, optimization_system: Optional[_EnhancedOptimizationSystem] = None):
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir
        self.optimization = optimization_system or _EnhancedOptimizationSystem()
        self.file_manager = FileManagementSystem()

    def optimize_target(self, file_path: str, reason: str) -> Dict[str, Any]:
        """Run an optimization pass on a target file."""
        try:
            self.logger.info(f"[AUTO-OPT] Optimizing {file_path} (reason={reason})")
            results = self.optimization.optimize_file_enhanced(file_path)
            applied = [r for r in results if getattr(r, "applied", False)]
            return {
                "success": len(applied) > 0,
                "applied_count": len(applied),
                "total_candidates": len(results),
            }
        except Exception as e:
            self.logger.error(f"[AUTO-OPT] Optimization failed: {e}")
            return {"success": False, "error": str(e)}


class AutonomousStrategyManager:
    """
    Main coordinator. Periodically:
      1) Collect metrics
      2) Decide actions
      3) Apply parameter and code/config changes
      4) Learn from outcomes and persist state
    """

    def __init__(self, base_dir: str, targets: Optional[List[str]] = None, interval_sec: int = 60):
        self.logger = logging.getLogger(__name__)
        self.base_dir = base_dir
        self.interval_sec = max(10, interval_sec)
        self.targets = targets or [
            os.path.join(base_dir, "GridbotBackup.py"),
            os.path.join(base_dir, "gridbot_websocket_server.py"),
        ]

        self.monitor = RealTimeMonitor(base_dir=self.base_dir)
        self.decision_engine = AdaptiveDecisionEngine()
        state_path = os.path.join(base_dir, "automation_logs", "strategy_state.json")
        self.learning = ContinuousLearningEngine(state_path)
        self.state = self.learning.load()
        self.auto_opt = AutoOptimizationPipeline(base_dir=self.base_dir)

        # Point runtime_config at a stable location in the workspace root
        try:
            if runtime_config is not None:
                runtime_config.set_config_path(os.path.join(self.base_dir, 'runtime_config.json'))
        except Exception:
            pass

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    # ------------- Lifecycle -------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            self.logger.info("[ASM] Already running")
            return
        self.logger.info("[ASM] Starting background strategy daemon")
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop, name="AutonomousStrategyDaemon", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self.logger.info("[ASM] Stopping background strategy daemon")
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    # ------------- Core Loop -------------
    def _run_loop(self):
        while not self._stop_evt.is_set():
            try:
                self.tick()
            except Exception as e:
                self.logger.error(f"[ASM] Tick error: {e}")
            self._stop_evt.wait(self.interval_sec)

    def tick(self) -> Dict[str, Any]:
        metrics = self.monitor.get_metrics()
        actions = self.decision_engine.decide(metrics, self.state)
        apply_report = self._apply_actions(actions)
        self.learning.update_from_outcome(self.state, metrics, actions)
        self.learning.save(self.state)
        return {"metrics": metrics, "actions": actions, "apply_report": apply_report}

    # ------------- Action Application -------------
    def _apply_actions(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        summary = {"param_updates": [], "optimizations": []}
        # Prepare in-memory parameter diffs for runtime propagation
        runtime_updates: Dict[str, Any] = {}

        for act in actions:
            atype = act.get("type")
            if atype == "adjust_param":
                key = act["key"]
                delta = act.get("delta", 0)
                cur = self.state.parameters.get(key, 0)
                new_val = cur + delta
                if "min" in act:
                    new_val = max(act["min"], new_val)
                if "max" in act:
                    new_val = min(act["max"], new_val)
                self.state.parameters[key] = new_val
                summary["param_updates"].append({"key": key, "old": cur, "new": new_val})
                runtime_updates[key] = new_val

            elif atype == "trigger_optimization":
                reason = act.get("reason", "decision")
                # Optimize primary targets
                for t in self.targets:
                    if os.path.exists(t):
                        res = self.auto_opt.optimize_target(t, reason=reason)
                        summary["optimizations"].append({"target": t, **res})

            elif atype == "explore_strategy_variant":
                variant = act.get("variant", "default")
                self.state.current_profile = variant
                summary["param_updates"].append({"key": "current_profile", "new": variant})

        # Push runtime parameter updates to file so long-running processes can hot-reload
        if runtime_updates and runtime_config is not None:
            try:
                if ok := runtime_config.update_parameters(runtime_updates):
                    self.logger.info(f"[ASM] Runtime parameters updated: {runtime_updates}")
                else:
                    self.logger.warning(f"[ASM] Failed to write runtime parameters: {runtime_updates}")
            except Exception as e:
                self.logger.warning(f"[ASM] Runtime parameter write error: {e}")

        return summary


__all__ = [
    "AutonomousStrategyManager",
    "StrategyState",
    "RealTimeMonitor",
    "AdaptiveDecisionEngine",
    "ContinuousLearningEngine",
    "AutoOptimizationPipeline",
]
