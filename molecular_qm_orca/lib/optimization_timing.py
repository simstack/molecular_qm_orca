from simstack.models.simple_table import SimpleTable, SimpleTableColumnType


def optimization_timing_table(snapshotter, freq_wall_s=None, freq_cpu_s=None):
    if freq_wall_s is not None and freq_cpu_s is None:
        raise ValueError("freq_cpu_s is required when freq_wall_s is set")
    if snapshotter is None:
        history = []
        opt_wall_s = None
        opt_cpu_s = None
    else:
        history = snapshotter.timing_history
        opt_wall_s = snapshotter.opt_wall_s
        opt_cpu_s = snapshotter.opt_cpu_s
    if not history and opt_wall_s is None and freq_wall_s is None:
        return None
    if opt_wall_s is not None and opt_cpu_s is None:
        raise ValueError("opt_cpu_s is required when opt_wall_s is set")
    table = SimpleTable(name="Optimization timing")
    table.add_column("metric", SimpleTableColumnType.STRING)
    table.add_column("step", SimpleTableColumnType.NUMBER)
    table.add_column("wall_time_s", SimpleTableColumnType.NUMBER)
    table.add_column("cpu_time_s", SimpleTableColumnType.NUMBER)
    walls = []
    cpus = []
    for row in history:
        step = row.get("step")
        wall_s = row.get("wall_time_s")
        cpu_s = row.get("cpu_time_s")
        if step is None:
            raise ValueError("timing_history step is required")
        if wall_s is None:
            raise ValueError(f"wall_time_s missing for optimization step {step}")
        if cpu_s is None:
            raise ValueError(f"cpu_time_s missing for optimization step {step}")
        table.add_row(
            {
                "metric": "iteration",
                "step": int(step),
                "wall_time_s": float(wall_s),
                "cpu_time_s": float(cpu_s),
            }
        )
        walls.append(float(wall_s))
        cpus.append(float(cpu_s))
    if walls:
        n_steps = len(walls)
        table.add_row(
            {
                "metric": "total",
                "step": None,
                "wall_time_s": sum(walls),
                "cpu_time_s": sum(cpus),
            }
        )
        table.add_row(
            {
                "metric": "mean",
                "step": None,
                "wall_time_s": sum(walls) / n_steps,
                "cpu_time_s": sum(cpus) / n_steps,
            }
        )
        table.add_row(
            {
                "metric": "min",
                "step": None,
                "wall_time_s": min(walls),
                "cpu_time_s": min(cpus),
            }
        )
        table.add_row(
            {
                "metric": "max",
                "step": None,
                "wall_time_s": max(walls),
                "cpu_time_s": max(cpus),
            }
        )
    if opt_wall_s is not None:
        table.add_row(
            {
                "metric": "optimize",
                "step": None,
                "wall_time_s": float(opt_wall_s),
                "cpu_time_s": float(opt_cpu_s),
            }
        )
    if freq_wall_s is not None:
        table.add_row(
            {
                "metric": "frequencies",
                "step": None,
                "wall_time_s": float(freq_wall_s),
                "cpu_time_s": float(freq_cpu_s),
            }
        )
    return table


def timing_metric_row(table, metric: str):
    if table is None:
        return None
    for row in getattr(table, "row", None) or []:
        if row.get("metric") == metric:
            return row
    return None


def attach_optimizer_timings(node_runner, snapshotter, freq_wall_s=None, freq_cpu_s=None) -> None:
    """Attach the per-iteration optimization timing table to the node result."""
    if node_runner is None:
        return
    table = optimization_timing_table(
        snapshotter, freq_wall_s=freq_wall_s, freq_cpu_s=freq_cpu_s
    )
    if table is None:
        return
    node_runner.optimization_timing = table
    n_steps = sum(1 for row in table.row if row.get("metric") == "iteration")
    chosen = timing_metric_row(table, "optimize") or timing_metric_row(table, "total")
    wall = None if chosen is None else chosen.get("wall_time_s")
    cpu = None if chosen is None else chosen.get("cpu_time_s")
    wall_text = "n/a" if wall is None else f"{float(wall):.2f}s"
    cpu_text = "n/a" if cpu is None else f"{float(cpu):.2f}s"
    message = f"Optimization timings: n_steps={n_steps}, wall={wall_text}, cpu={cpu_text}"
    freq_row = timing_metric_row(table, "frequencies")
    if freq_row is not None:
        freq_wall = freq_row.get("wall_time_s")
        freq_cpu = freq_row.get("cpu_time_s")
        freq_wall_text = "n/a" if freq_wall is None else f"{float(freq_wall):.2f}s"
        freq_cpu_text = "n/a" if freq_cpu is None else f"{float(freq_cpu):.2f}s"
        message = (
            f"{message}; frequencies wall={freq_wall_text}, cpu={freq_cpu_text}"
        )
    node_runner.info(message)
