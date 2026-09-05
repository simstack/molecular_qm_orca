"""Live ORCA monitoring: geometry-cycle logs, chart artifacts, and file attachments."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from odmantic import ObjectId

from simstack.core.context import context
from simstack.models.charts_artifact import (
    AGChartAxisConfig,
    AGChartTitleConfig,
    AGLineSeriesConfig,
    ChartArtifactModel,
)
from simstack.models.files import FileStack

from .process_heartbeat import ProcessHeartbeat

logger = logging.getLogger(__name__)

HEARTBEAT_LOG = "heartbeat.log"
HEARTBEAT_INTERVAL_S = 1800.0
OPT_CHART_INTERVAL = 10
OPT_CHART_STEPS = 20
ORCA_RUN_LOG = "orca_run.log"

_CYCLE_HEADER = re.compile(
    r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)",
    re.IGNORECASE,
)
_ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)")
_CONV_ITEM = re.compile(
    r"^\s*(Energy change|RMS gradient|MAX gradient|RMS step|MAX step)\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)",
    re.IGNORECASE,
)
_CYCLE_TIME = re.compile(
    r"Total time for this optimization cycle:\s+(.+)$",
    re.IGNORECASE,
)
_TOTAL_RUN_TIME = re.compile(r"TOTAL RUN TIME:\s+(.+)$", re.IGNORECASE)
_SUM_INDIVIDUAL = re.compile(
    r"Sum of individual times\s+\.+\s+([+-]?\d+(?:\.\d+)?)\s+sec",
    re.IGNORECASE,
)
_MODULE_TIME = re.compile(
    r"^\s*(.+?)\s+\.+\s+([+-]?\d+(?:\.\d+)?)\s+sec",
)
_CART_GRAD_LINE = re.compile(
    r"^\s*\d+\s+[A-Za-z]{1,3}\s*:\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[EeDd][+-]?\d+)?)"
)
_DURATION_PARTS = re.compile(
    r"(?:(\d+)\s+days?)?\s*"
    r"(?:(\d+)\s+hours?)?\s*"
    r"(?:(\d+)\s+min(?:utes)?)?\s*"
    r"(?:(\d+(?:\.\d+)?)\s+sec(?:onds)?)?\s*"
    r"(?:(\d+)\s+msec)?",
    re.IGNORECASE,
)


def parse_orca_duration(text: str) -> float:
    """Parse an ORCA duration string into seconds.

    Accepts ``TOTAL RUN TIME`` / cycle-time strings such as
    ``0 days 0 hours 1 minutes 23 seconds 456 msec`` or ``12.345 sec``.
    """
    if text is None:
        raise ValueError("duration text is required")
    stripped = str(text).strip()
    if not stripped:
        raise ValueError("duration text is empty")
    match = _DURATION_PARTS.search(stripped)
    if match is None or not any(match.groups()):
        raise ValueError(f"unrecognized ORCA duration: {stripped!r}")
    days, hours, minutes, seconds, msec = match.groups()
    total = 0.0
    if days:
        total += int(days) * 86400
    if hours:
        total += int(hours) * 3600
    if minutes:
        total += int(minutes) * 60
    if seconds:
        total += float(seconds)
    if msec:
        total += int(msec) / 1000.0
    return total


def _fortran_float(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def parse_orca_opt_cycles(content: str) -> list[dict]:
    """Parse geometry-optimization cycles from ``orca.out`` text.

    Each cycle dict has ``step``, and optionally ``energy``, ``grad_norm``,
    and ``cycle_time_s`` when those values appear in the output.
    """
    if content is None:
        raise ValueError("content is required")
    lines = content.splitlines()
    cycles: dict[int, dict] = {}
    current_step = None
    in_convergence = False
    in_cart_grad = False
    cart_components: list[float] = []

    def _cycle(step: int) -> dict:
        row = cycles.setdefault(step, {"step": step})
        return row

    for raw in lines:
        header = _CYCLE_HEADER.search(raw)
        if header:
            current_step = int(header.group(1))
            _cycle(current_step)
            in_convergence = False
            in_cart_grad = False
            cart_components = []
            continue

        energy_match = _ENERGY.search(raw)
        if energy_match and current_step is not None:
            _cycle(current_step)["energy"] = float(energy_match.group(1))
            continue

        if "Geometry convergence" in raw:
            in_convergence = True
            in_cart_grad = False
            continue
        if in_convergence:
            item = _CONV_ITEM.match(raw)
            if item and current_step is not None:
                name = item.group(1).lower()
                value = _fortran_float(item.group(2))
                row = _cycle(current_step)
                if name == "rms gradient":
                    row["grad_norm"] = value
                elif name == "max gradient":
                    row["max_gradient"] = value
                elif name == "energy change":
                    row["energy_change"] = value
                continue
            if raw.strip().startswith("-----") and any(
                key in cycles.get(current_step, {}) for key in ("grad_norm", "max_gradient")
            ):
                in_convergence = False
                continue

        if "CARTESIAN GRADIENT" in raw:
            in_cart_grad = True
            cart_components = []
            continue
        if in_cart_grad:
            grad = _CART_GRAD_LINE.match(raw)
            if grad:
                cart_components.extend(_fortran_float(p) for p in grad.groups())
                continue
            if cart_components and current_step is not None:
                row = _cycle(current_step)
                if "grad_norm" not in row:
                    row["grad_norm"] = math.sqrt(sum(c * c for c in cart_components))
                in_cart_grad = False

        cycle_time = _CYCLE_TIME.search(raw)
        if cycle_time and current_step is not None:
            _cycle(current_step)["cycle_time_s"] = parse_orca_duration(cycle_time.group(1))

    return [cycles[step] for step in sorted(cycles)]


def parse_orca_run_timings(content: str) -> dict:
    """Parse overall ORCA TIMINGS / TOTAL RUN TIME from ``orca.out`` text."""
    if content is None:
        raise ValueError("content is required")
    total_run_s = None
    sum_individual_s = None
    freq_s = None
    for raw in content.splitlines():
        total = _TOTAL_RUN_TIME.search(raw)
        if total:
            total_run_s = parse_orca_duration(total.group(1))
            continue
        summed = _SUM_INDIVIDUAL.search(raw)
        if summed:
            sum_individual_s = float(summed.group(1))
            continue
        module = _MODULE_TIME.match(raw)
        if module:
            name = module.group(1).strip().lower()
            if "freq" in name or "vibrational" in name:
                freq_s = float(module.group(2))
    return {
        "total_run_s": total_run_s,
        "sum_individual_s": sum_individual_s,
        "freq_s": freq_s,
    }


def task_id_from_kwargs(kwargs: dict) -> str:
    task_id = kwargs.get("task_id")
    if task_id is None:
        node_runner = None if not kwargs else kwargs.get("node_runner")
        task_id = getattr(node_runner, "task_id", None)
    return "" if task_id is None else str(task_id)


def task_parent_id(kwargs: dict):
    task_id = task_id_from_kwargs(kwargs)
    if not task_id:
        return None
    try:
        return ObjectId(str(task_id))
    except Exception:
        return None


def _get_db():
    try:
        return context.db
    except RuntimeError:
        return None


def _run_async(coro):
    """Run ``coro`` from a monitor thread that may already be inside an event loop."""
    try:
        import nest_asyncio

        nest_asyncio.apply()
    except Exception:
        pass
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    try:
        return asyncio.run(coro)
    except RuntimeError:
        return loop.run_until_complete(coro)


def opt_line_chart(data, y_key, title, y_label, parent_id, existing=None):
    series = AGLineSeriesConfig(
        type="line",
        xKey="step",
        yKey=y_key,
        title=y_label,
        data=data,
        marker={"enabled": False},
    )
    axes = [
        AGChartAxisConfig(type="number", position="bottom", title="Optimization step"),
        AGChartAxisConfig(type="number", position="left", title=y_label),
    ]
    if existing is not None:
        existing.data = data
        existing.title = AGChartTitleConfig(text=title)
        existing.series = [series]
        existing.axes = axes
        existing.parent_id = parent_id
        return existing
    return ChartArtifactModel(
        parent_id=parent_id,
        data=data,
        title=AGChartTitleConfig(text=title),
        series=[series],
        axes=axes,
    )


async def persist_opt_charts(energy_data, grad_data, kwargs, existing=(None, None)):
    node_runner = None if not kwargs else kwargs.get("node_runner")
    parent_id = task_parent_id(kwargs)
    if parent_id is None:
        if node_runner is not None:
            node_runner.warning("Skipping optimization charts: missing task_id")
        return existing
    db = _get_db()
    if db is None:
        return existing
    energy_chart = opt_line_chart(
        list(energy_data)[-OPT_CHART_STEPS:],
        "energy",
        "ORCA optimization energy",
        "Energy (Ha)",
        parent_id,
        existing[0],
    )
    grad_chart = opt_line_chart(
        list(grad_data)[-OPT_CHART_STEPS:],
        "grad_norm",
        "ORCA optimization gradient norm",
        "|g| (Ha/Bohr)",
        parent_id,
        existing[1],
    )
    try:
        await db.save(energy_chart)
        await db.save(grad_chart)
    except Exception as exc:
        if node_runner is not None:
            node_runner.warning(f"Failed to store optimization charts: {exc}")
        else:
            logger.warning("Failed to store optimization charts: %s", exc)
        return existing
    if node_runner is not None and energy_data:
        node_runner.info(
            f"Saved optimization charts at step {energy_data[-1]['step']} "
            f"(task_id={parent_id})"
        )
    return energy_chart, grad_chart


def artifact_names(node_runner) -> set:
    names = set()
    if node_runner is None:
        return names
    for fs in list(getattr(node_runner, "files", []) or []) + list(
        getattr(node_runner, "info_files", []) or []
    ):
        name = getattr(fs, "name", None)
        if name:
            names.add(name)
    return names


def append_info_file(node_runner, path: Path, *, in_memory: bool):
    """Attach a human-readable log/info file to ``node_runner.info_files`` only."""
    if node_runner is None:
        return
    path = Path(path)
    if not path.is_file():
        return
    if path.name in artifact_names(node_runner):
        return
    task_id = getattr(node_runner, "task_id", "") or ""
    fs = FileStack.from_local_file(
        path,
        in_memory=in_memory,
        is_hashable=True,
        secure_source=True,
        task_id=task_id,
    )
    node_runner.info_files.append(fs)
    node_runner.info(f"Attached ORCA info file: {path.name}")


def log_energy_gradient_summary(node_runner, monitor, output_path: Path | None):
    if node_runner is None:
        return
    node_runner.log("ORCA run summary")
    if monitor is not None and monitor.energy_history:
        node_runner.log(
            f"Optimization steps recorded: {len(monitor.energy_history)} "
            f"(geom_iter={monitor.geom_iter})"
        )
        for energy_row, grad_row in zip(monitor.energy_history, monitor.grad_history):
            stamp = energy_row.get("timestamp") or ""
            step = energy_row.get("step")
            energy = energy_row.get("energy")
            grad_norm = grad_row.get("grad_norm") if grad_row else None
            stamp_bit = f"{stamp} " if stamp else ""
            energy_bit = f"{float(energy):.12f} Ha" if energy is not None else "unavailable"
            grad_bit = (
                f"{float(grad_norm):.6e} Ha/Bohr" if grad_norm is not None else "unavailable"
            )
            node_runner.log(
                f"{stamp_bit}step {step}: energy={energy_bit}, |g|={grad_bit}"
            )
    elif monitor is not None:
        node_runner.log(
            f"No optimization energy/gradient history (geom_iter={monitor.geom_iter})"
        )
    if output_path is not None and Path(output_path).is_file():
        timings = parse_orca_run_timings(
            Path(output_path).read_text(encoding="utf-8", errors="replace")
        )
        if timings["total_run_s"] is not None:
            node_runner.log(f"ORCA TOTAL RUN TIME: {timings['total_run_s']:.3f}s")
        if timings["sum_individual_s"] is not None:
            node_runner.log(
                f"ORCA sum of individual times: {timings['sum_individual_s']:.3f}s"
            )


class OrcaRunMonitor:
    """Tail ``orca.out`` while ORCA runs: heartbeat, cycle logs, charts, timings."""

    def __init__(
        self,
        kwargs: dict,
        output_path: str | Path = "orca.out",
        interval: int = OPT_CHART_INTERVAL,
        poll_s: float = 5.0,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
    ):
        if interval is None:
            raise ValueError("interval is required")
        if poll_s is None:
            raise ValueError("poll_s is required")
        if float(poll_s) <= 0:
            raise ValueError("poll_s must be positive")
        if int(interval) < 1:
            raise ValueError("interval must be >= 1")
        self.kwargs = kwargs or {}
        self.output_path = Path(output_path)
        self.interval = int(interval)
        self.poll_s = float(poll_s)
        self.heartbeat_interval_s = float(heartbeat_interval_s)
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        self.energy_history: list[dict] = []
        self.grad_history: list[dict] = []
        self.timing_history: list[dict] = []
        self.opt_wall_s = None
        self.opt_cpu_s = None
        self.freq_wall_s = None
        self.freq_cpu_s = None
        self.geom_iter = 0
        self.charts = (None, None)
        self._chart_steps: set[int] = set()
        self._logged_steps: set[int] = set()
        self._timed_steps: set[int] = set()
        self._stop = threading.Event()
        self._thread = None
        self._heartbeat = None
        self._wall_start = None
        self._lock = threading.Lock()

    def _node_runner(self):
        return self.kwargs.get("node_runner")

    def start(self):
        if self._thread is not None:
            return
        self._wall_start = time.monotonic()
        node_runner = self._node_runner()
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        start_msg = (
            f"{stamp} Starting ORCA calculation "
            f"(heartbeat every {self.heartbeat_interval_s:g}s)"
        )
        if node_runner is not None:
            node_runner.log(start_msg)
            node_runner.info(start_msg)
        else:
            logger.info(start_msg)
        task_id = "" if node_runner is None else str(getattr(node_runner, "task_id", "") or "")
        heartbeat = ProcessHeartbeat(
            HEARTBEAT_LOG,
            "ORCA calculation",
            interval_s=self.heartbeat_interval_s,
            task_id=task_id,
            extra_paths=[ORCA_RUN_LOG],
        )
        heartbeat.start()
        self._heartbeat = heartbeat
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="orca-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            thread.join(timeout=max(self.poll_s * 2, 5.0))
        heartbeat = self._heartbeat
        self._heartbeat = None
        if heartbeat is not None:
            heartbeat.stop()
        self.consume_output()
        if self._wall_start is not None:
            wall_s = time.monotonic() - self._wall_start
            self._wall_start = None
            node_runner = self._node_runner()
            if node_runner is not None:
                node_runner.info(f"ORCA subprocess wall time: {wall_s:.2f}s")
            self._apply_run_timings(wall_s)
        self._flush_opt_charts()

    def _poll_loop(self):
        while not self._stop.wait(self.poll_s):
            try:
                self.consume_output()
            except Exception as exc:
                logger.warning("ORCA monitor poll failed: %s", exc)

    def consume_output(self):
        if not self.output_path.is_file():
            return
        content = self.output_path.read_text(encoding="utf-8", errors="replace")
        cycles = parse_orca_opt_cycles(content)
        for cycle in cycles:
            self._record_cycle(cycle)

    def _record_cycle(self, cycle: dict):
        step = int(cycle["step"])
        energy = cycle.get("energy")
        grad_norm = cycle.get("grad_norm")
        cycle_time_s = cycle.get("cycle_time_s")
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            self.geom_iter = max(self.geom_iter, step)
            newly_logged = step not in self._logged_steps
            if newly_logged:
                self._logged_steps.add(step)
            if energy is not None and grad_norm is not None and step not in self._chart_steps:
                self._chart_steps.add(step)
                self.energy_history.append(
                    {"step": step, "energy": energy, "timestamp": stamp}
                )
                self.grad_history.append(
                    {"step": step, "grad_norm": grad_norm, "timestamp": stamp}
                )
                should_flush = step % self.interval == 0
            else:
                should_flush = False
            if (
                cycle_time_s is not None
                and step not in self._timed_steps
            ):
                self._timed_steps.add(step)
                self.timing_history.append(
                    {
                        "step": step,
                        "wall_time_s": float(cycle_time_s),
                        "cpu_time_s": float(cycle_time_s),
                        "timestamp": stamp,
                        "energy": energy,
                        "grad_norm": grad_norm,
                    }
                )
        if newly_logged:
            if energy is None or grad_norm is None:
                msg = f"{stamp} Optimization step {step}: energy/gradient unavailable"
            else:
                msg = (
                    f"{stamp} Optimization step {step}: "
                    f"energy={float(energy):.12f} Ha, |g|={float(grad_norm):.6e} Ha/Bohr"
                )
            if cycle_time_s is not None:
                msg = f"{msg}, wall={float(cycle_time_s):.2f}s, cpu={float(cycle_time_s):.2f}s"
            node_runner = self._node_runner()
            if node_runner is not None:
                node_runner.log(msg)
                node_runner.info(msg)
            else:
                logger.info(msg)
        if should_flush:
            self._flush_opt_charts()

    def _apply_run_timings(self, wall_s: float):
        if not self.output_path.is_file():
            return
        parsed = parse_orca_run_timings(
            self.output_path.read_text(encoding="utf-8", errors="replace")
        )
        cpu_s = parsed["sum_individual_s"]
        if cpu_s is None:
            return
        self.opt_wall_s = float(wall_s)
        self.opt_cpu_s = float(cpu_s)
        if parsed["freq_s"] is not None:
            self.freq_wall_s = float(parsed["freq_s"])
            self.freq_cpu_s = float(parsed["freq_s"])

    def _flush_opt_charts(self):
        if not self.energy_history:
            return
        try:
            saved = _run_async(
                persist_opt_charts(
                    self.energy_history,
                    self.grad_history,
                    self.kwargs,
                    self.charts,
                )
            )
        except Exception as exc:
            node_runner = self._node_runner()
            if node_runner is not None:
                node_runner.warning(f"Failed to store optimization charts: {exc}")
            else:
                logger.warning("Failed to store optimization charts: %s", exc)
            return
        if saved is not None:
            self.charts = saved
