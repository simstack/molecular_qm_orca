import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def initialized_context():
    """These tests do not need a Simstack context or simstack.toml."""
    yield

from molecular_qm_orca.lib.opt_monitor import (
    HEARTBEAT_LOG,
    OrcaRunMonitor,
    append_info_file,
    parse_orca_duration,
    parse_orca_opt_cycles,
    parse_orca_run_timings,
)
from molecular_qm_orca.lib.optimization_timing import (
    attach_optimizer_timings,
    optimization_timing_table,
)
from molecular_qm_orca.lib.process_heartbeat import ProcessHeartbeat
from molecular_qm_orca.nodes.orca import ORCA_INFO_FILES, _collect_existing_orca_info_files


_OPT_OUT = """
          *****************************************************************
          *                    GEOMETRY OPTIMIZATION CYCLE   1            *
          *****************************************************************

          FINAL SINGLE POINT ENERGY      -76.434219876543

                              .--------------------.
          ----------------------|Geometry convergence|-------------------------
          Item                value                   Tolerance       Converged
          ---------------------------------------------------------------------
          Energy change       0.0000000000            0.0000050000      YES
          RMS gradient        0.0023456789            0.0001000000      NO
          MAX gradient        0.0065432109            0.0003000000      NO
          RMS step            0.0000000000            0.0020000000      YES
          MAX step            0.0000000000            0.0040000000      YES
          ---------------------------------------------------------------------

          Total time for this optimization cycle: 0 days 0 hours 0 min 12.345 sec

          *****************************************************************
          *                    GEOMETRY OPTIMIZATION CYCLE   2            *
          *****************************************************************

          FINAL SINGLE POINT ENERGY      -76.435000000000

          ----------------------|Geometry convergence|-------------------------
          Item                value                   Tolerance       Converged
          ---------------------------------------------------------------------
          Energy change      -0.0007801234            0.0000050000      NO
          RMS gradient        0.0001234567            0.0001000000      NO
          MAX gradient        0.0003456789            0.0003000000      NO
          ---------------------------------------------------------------------

          Total time for this optimization cycle: 0 days 0 hours 0 min 8.000 sec

Timings for individual modules:

Sum of individual times         ...       20.345 sec (=   0.006 hours)
                                  Freq                    ...        5.678 sec

TOTAL RUN TIME: 0 days 0 hours 0 minutes 21 seconds 123 msec
"""


def test_parse_orca_duration_total_run_time():
    assert parse_orca_duration("0 days 0 hours 0 minutes 21 seconds 123 msec") == pytest.approx(
        21.123
    )
    assert parse_orca_duration("0 days 0 hours 0 min 12.345 sec") == pytest.approx(12.345)
    assert parse_orca_duration("1 hours 2 minutes 3 seconds") == pytest.approx(3723.0)


def test_parse_orca_duration_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        parse_orca_duration("   ")
    with pytest.raises(ValueError, match="required"):
        parse_orca_duration(None)
    with pytest.raises(ValueError, match="unrecognized"):
        parse_orca_duration("not a duration")


def test_parse_orca_opt_cycles():
    cycles = parse_orca_opt_cycles(_OPT_OUT)
    assert [row["step"] for row in cycles] == [1, 2]
    assert cycles[0]["energy"] == pytest.approx(-76.434219876543)
    assert cycles[0]["grad_norm"] == pytest.approx(0.0023456789)
    assert cycles[0]["cycle_time_s"] == pytest.approx(12.345)
    assert cycles[1]["energy"] == pytest.approx(-76.435)
    assert cycles[1]["grad_norm"] == pytest.approx(0.0001234567)
    assert cycles[1]["cycle_time_s"] == pytest.approx(8.0)


def test_parse_orca_opt_cycles_cartesian_gradient_fallback():
    text = """
          GEOMETRY OPTIMIZATION CYCLE   1
          FINAL SINGLE POINT ENERGY      -1.0
          CARTESIAN GRADIENT
          -----------------
   1   O   :    0.000000000    0.000000000    0.030000000
   2   H   :    0.000000000    0.040000000    0.000000000
   3   H   :    0.000000000   -0.040000000    0.000000000
          Norm of the cartesian gradient: whatever
"""
    cycles = parse_orca_opt_cycles(text)
    assert len(cycles) == 1
    assert cycles[0]["grad_norm"] == pytest.approx(math.sqrt(0.03**2 + 0.04**2 + 0.04**2))


def test_parse_orca_opt_cycles_requires_content():
    with pytest.raises(ValueError, match="required"):
        parse_orca_opt_cycles(None)


def test_parse_orca_run_timings():
    parsed = parse_orca_run_timings(_OPT_OUT)
    assert parsed["total_run_s"] == pytest.approx(21.123)
    assert parsed["sum_individual_s"] == pytest.approx(20.345)
    assert parsed["freq_s"] == pytest.approx(5.678)


def test_optimization_timing_table_has_iteration_and_summary_rows():
    snap = SimpleNamespace(
        timing_history=[
            {"step": 1, "wall_time_s": 2.0, "cpu_time_s": 1.5},
            {"step": 2, "wall_time_s": 4.0, "cpu_time_s": 3.0},
        ],
        opt_wall_s=7.0,
        opt_cpu_s=5.0,
    )
    table = optimization_timing_table(snap)
    assert table.name == "Optimization timing"
    assert table.row[0]["metric"] == "iteration"
    by_metric = {row["metric"]: row for row in table.row if row["metric"] != "iteration"}
    assert by_metric["total"]["wall_time_s"] == 6.0
    assert by_metric["mean"]["cpu_time_s"] == 2.25
    assert by_metric["optimize"]["wall_time_s"] == 7.0


def test_optimization_timing_table_includes_frequencies_row():
    table = optimization_timing_table(None, freq_wall_s=4.5, freq_cpu_s=9.0)
    assert table.row[0]["metric"] == "frequencies"
    assert table.row[0]["wall_time_s"] == 4.5


def test_optimization_timing_table_requires_cpu_when_wall_set():
    snap = SimpleNamespace(timing_history=[], opt_wall_s=1.0, opt_cpu_s=None)
    with pytest.raises(ValueError, match="opt_cpu_s"):
        optimization_timing_table(snap)
    with pytest.raises(ValueError, match="freq_cpu_s"):
        optimization_timing_table(None, freq_wall_s=1.0)


def test_attach_optimizer_timings_logs_summary():
    node_runner = MagicMock()
    snap = SimpleNamespace(
        timing_history=[{"step": 1, "wall_time_s": 2.0, "cpu_time_s": 1.5}],
        opt_wall_s=2.0,
        opt_cpu_s=1.5,
    )
    attach_optimizer_timings(node_runner, snap)
    assert node_runner.optimization_timing.row[0]["metric"] == "iteration"
    messages = [call.args[0] for call in node_runner.info.call_args_list]
    assert any("Optimization timings: n_steps=1" in msg for msg in messages)


def test_monitor_consume_output_logs_steps(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "orca.out").write_text(_OPT_OUT, encoding="utf-8")
    node_runner = MagicMock()
    monitor = OrcaRunMonitor({"node_runner": node_runner}, poll_s=0.1)
    monitor.consume_output()
    messages = [call.args[0] for call in node_runner.info.call_args_list]
    step_logs = [msg for msg in messages if "Optimization step " in msg]
    assert len(step_logs) == 2
    assert "energy=-76.434219876543 Ha" in step_logs[0]
    assert "|g|=2.345679e-03 Ha/Bohr" in step_logs[0]
    assert "wall=12.35s" in step_logs[0]
    logged = [call.args[0] for call in node_runner.log.call_args_list]
    assert step_logs == [msg for msg in logged if "Optimization step " in msg]
    assert [row["step"] for row in monitor.timing_history] == [1, 2]
    assert [row["step"] for row in monitor.energy_history] == [1, 2]
    monitor.consume_output()
    assert len([msg for msg in node_runner.info.call_args_list if "Optimization step " in msg.args[0]]) == 2


def test_monitor_apply_run_timings_requires_cpu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "orca.out").write_text(
        "TOTAL RUN TIME: 0 days 0 hours 0 minutes 5 seconds 0 msec\n",
        encoding="utf-8",
    )
    monitor = OrcaRunMonitor({})
    monitor._apply_run_timings(5.0)
    assert monitor.opt_wall_s is None
    assert monitor.opt_cpu_s is None

    (tmp_path / "orca.out").write_text(_OPT_OUT, encoding="utf-8")
    monitor._apply_run_timings(21.5)
    assert monitor.opt_wall_s == pytest.approx(21.5)
    assert monitor.opt_cpu_s == pytest.approx(20.345)
    assert monitor.freq_wall_s == pytest.approx(5.678)
    assert monitor.freq_cpu_s == pytest.approx(5.678)


def test_append_info_file_skips_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "orca.out"
    target.write_text("out", encoding="utf-8")
    existing = SimpleNamespace(name="orca.out")
    node_runner = SimpleNamespace(
        files=[],
        info_files=[existing],
        task_id="task-1",
        info=MagicMock(),
    )
    with patch(
        "molecular_qm_orca.lib.opt_monitor.FileStack.from_local_file"
    ) as mocked:
        append_info_file(node_runner, target, in_memory=True)
    mocked.assert_not_called()
    assert node_runner.info_files == [existing]


def test_collect_info_files_includes_heartbeat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "orca.out").write_text("out", encoding="utf-8")
    (tmp_path / HEARTBEAT_LOG).write_text("beat\n", encoding="utf-8")
    node_runner = SimpleNamespace(files=[], info_files=[], task_id="t", info=MagicMock(), warning=MagicMock())
    with patch(
        "molecular_qm_orca.lib.opt_monitor.FileStack.from_local_file",
        side_effect=lambda path, **kwargs: SimpleNamespace(name=Path(path).name),
    ):
        _collect_existing_orca_info_files(node_runner)
    names = {fs.name for fs in node_runner.info_files}
    assert "orca.out" in names
    assert HEARTBEAT_LOG in names
    assert HEARTBEAT_LOG in ORCA_INFO_FILES


def test_process_heartbeat_appends_until_stopped(tmp_path):
    path = tmp_path / "heartbeat.log"
    extra = tmp_path / "orca_run.log"
    heartbeat = ProcessHeartbeat(
        str(path),
        "ORCA calculation",
        interval_s=0.2,
        task_id="abc",
        extra_paths=[str(extra)],
    )
    heartbeat.start()
    try:
        deadline = time.time() + 5
        while time.time() < deadline and not path.exists():
            time.sleep(0.05)
        time.sleep(0.45)
    finally:
        heartbeat.stop()
    text = path.read_text(encoding="utf-8")
    assert "ORCA calculation" in text
    assert "still running" in text
    assert extra.exists()
    assert "ORCA calculation" in extra.read_text(encoding="utf-8")


def test_process_heartbeat_hides_windows_console():
    heartbeat = ProcessHeartbeat("hb.log", "ORCA calculation", interval_s=1.0)
    with patch("molecular_qm_orca.lib.process_heartbeat.subprocess.Popen") as popen:
        popen.return_value = MagicMock()
        heartbeat.start()
    kwargs = popen.call_args.kwargs
    assert kwargs["stdout"] is not None
    assert kwargs["start_new_session"] is True
    if sys.platform == "win32":
        import subprocess

        startupinfo = kwargs["startupinfo"]
        assert startupinfo.dwFlags & subprocess.STARTF_USESHOWWINDOW
        assert startupinfo.wShowWindow == subprocess.SW_HIDE
    else:
        assert "startupinfo" not in kwargs


def test_monitor_rejects_undefined_intervals():
    with pytest.raises(ValueError, match="poll_s"):
        OrcaRunMonitor({}, poll_s=None)
    with pytest.raises(ValueError, match="positive"):
        OrcaRunMonitor({}, poll_s=0)
    with pytest.raises(ValueError, match="interval"):
        OrcaRunMonitor({}, interval=0)
