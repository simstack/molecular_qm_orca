"""Slurm CPU count aligned with simstack.core.run_docker.docker_cpu_limit."""


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 1 else None


def _slurm_task_count(slurm: object) -> int:
    tasks = _positive_int(getattr(slurm, "tasks", None))
    if tasks is not None:
        return tasks
    tasks_per_node = _positive_int(getattr(slurm, "tasks_per_node", None))
    if tasks_per_node is not None:
        return tasks_per_node
    return 1


def slurm_cpu_count(slurm: object | None) -> int | None:
    """Effective CPU count: (cpus_per_task or 1) * task count from tasks or tasks_per_node."""
    if slurm is None:
        return None

    cpus_per_task = _positive_int(getattr(slurm, "cpus_per_task", None))
    tasks = _positive_int(getattr(slurm, "tasks", None))
    tasks_per_node = _positive_int(getattr(slurm, "tasks_per_node", None))
    if cpus_per_task is None and tasks is None and tasks_per_node is None:
        return None

    return (cpus_per_task or 1) * _slurm_task_count(slurm)
