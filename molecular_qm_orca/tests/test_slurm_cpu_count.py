from simstack.models.parameters import Parameters, SlurmParameters

from molecular_qm_orca.lib.orca_input import orca_input_factory
from molecular_qm_orca.lib.slurm_cpu_count import slurm_cpu_count
from molecular_qm_orca.tests.test_orca_qm_input import _qm_input


def test_slurm_cpu_count_tasks_times_cpus_per_task():
    slurm = SlurmParameters(tasks=12, cpus_per_task=1, tasks_per_node=1)
    assert slurm_cpu_count(slurm) == 12


def test_slurm_cpu_count_cpus_per_task_only():
    slurm = SlurmParameters(cpus_per_task=4)
    assert slurm_cpu_count(slurm) == 4


def test_slurm_cpu_count_unset_returns_none():
    assert slurm_cpu_count(SlurmParameters()) is None


def test_cores_block_emits_pal_for_tasks():
    parent = Parameters(
        slurm_parameters=SlurmParameters(
            tasks=12,
            cpus_per_task=1,
            tasks_per_node=1,
            mem="12G",
        )
    )
    gen = orca_input_factory(_qm_input(), parent_parameters=parent)
    assert "nprocs 12" in gen.cores_block


def test_memory_block_splits_mem_by_slurm_cpu_count():
    parent = Parameters(
        slurm_parameters=SlurmParameters(
            tasks=12,
            cpus_per_task=1,
            tasks_per_node=1,
            mem="12G",
        )
    )
    gen = orca_input_factory(_qm_input(), parent_parameters=parent)
    assert gen.memory_block == "%maxcore 1124\n"
