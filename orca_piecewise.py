"""Piecewise ORCA workflow helpers and nodes.

This module contains the building blocks used to construct ORCA workflows
in smaller steps (``input_file_only``, ``run_only``, ``collect_results``)
as well as combined convenience nodes that mirror the behaviour of the
monolithic :func:`orca` / :func:`orca_jinja` nodes.

The goal is to keep :mod:`orca.orca` focused on the high-level nodes
while centralising shared parsing and piecewise logic here.
"""

#from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Iterable

from .orca_absorption_spectrum_parser import (
    parse_orca_absorption_spectrum,
)
from .orca_excited_states_parser import (
    parse_orca_excited_states,
)
from .orca_frequency_parser import (
    parse_vibrational_frequencies,
    parse_normal_modes,
    parse_ir_spectrum,
)
from .orca_mayer_parser import (
    parse_mayer_analysis,
)

from molecular_qm_models import QMInput
from molecular_qm_models.qm_result import QMResult
from molecular_qm_models import QMResult_elprop
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.core.simstack_result import SimstackResult
from simstack.models import Parameters
from simstack.models.file_list import FileListIO, FileListModel
from simstack.models.files import FileStack
from simstack.models.parameters import SlurmParameters
from .pyorca import OrcaRun, OrcaInput

logger = logging.getLogger("OrcaNode")


# Default node parameters (mirrors applications.electronic_structure.orca.orca)
slurm_parameters = SlurmParameters()
parameters = Parameters(resource="int-nano", queue="slurm-queue", slurm_parameters=slurm_parameters)


def _build_orca_command_from_config(resource: str, program_name: str = "orca") -> str | None:
    """Build the ORCA run command from ``config.toml`` for the given resource.

    This helper is shared between the monolithic and piecewise ORCA nodes as
    well as their Jinja/config-driven counterparts.
    """

    try:
        config_path = context.config.project_root / "config.toml"
    except Exception:
        # Older Simstack versions may not expose project_root on the config
        return None

    if not config_path.exists():
        logger.warning("config.toml not found at %s; falling back to legacy ORCA command", config_path)
        return None

    try:
        # ``tomllib`` import is handled in orca.orca; we keep a local, minimal
        # parser here via the same logic to avoid another dependency.
        try:  # Python 3.11+
            import tomllib  # type: ignore[attr-defined]
        except ImportError:  # pragma: no cover - fallback for older interpreters
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:  # pragma: no cover - defensive
        logger.error("Failed to read config.toml: %s", e)
        return None

    res_cfg = cfg.get(resource)
    if not isinstance(res_cfg, dict):
        logger.warning("No section '[%s]' in config.toml; falling back to legacy ORCA command", resource)
        return None

    prog_root = res_cfg.get("program") or {}
    prog_cfg = prog_root.get(program_name)
    if not isinstance(prog_cfg, dict):
        logger.warning(
            "No program '%s' configured under '[%s.program]' in config.toml; "
            "falling back to legacy ORCA command",
            program_name,
            resource,
        )
        return None

    scripts: list[str] = []

    # Optional explicit shell snippets
    scripts_cfg = prog_cfg.get("scripts") or []
    if isinstance(scripts_cfg, (list, tuple)):
        scripts.extend(str(s).strip() for s in scripts_cfg if str(s).strip())

    # Optional environment modules
    env_mods = prog_cfg.get("environment_modules") or []
    if isinstance(env_mods, (list, tuple)):
        for mod in env_mods:
            mod_str = str(mod).strip()
            if mod_str:
                scripts.append(f"module load {mod_str}")

    run_command = prog_cfg.get("run_command")
    if not isinstance(run_command, str) or not run_command.strip():
        logger.warning(
            "[%%s.program.%s] in config.toml has no valid 'run_command'; "
            "falling back to legacy ORCA command",
            program_name,
            resource,
        )
        return None

    scripts.append(run_command.strip())
    return " && ".join(scripts)


def _log_elprop_basic(orca_run: OrcaRun, node_runner: NodeRunner) -> None:
    """Log a few basic electronic-property quantities for debugging.

    This is shared between monolithic and piecewise nodes so that logs are
    comparable regardless of the workflow structure.
    """

    try:
        dipole_val = getattr(orca_run, "dipole", None)
        dipole_moment_val = getattr(orca_run, "dipole_moment", None)
        node_runner.info(
            f"ORCA result electronic properties: dipole={dipole_val}, "
            f"dipole_moment={dipole_moment_val}"
        )

        hyper_info = getattr(orca_run, "_hyperpolarizability", None)
        node_runner.info(
            f"ORCA hyperpolarizability mapping present: {hyper_info is not None}"
        )
    except Exception as e_el:  # pragma: no cover - debug logging only
        node_runner.warning(
            f"Failed to log ORCA/QMResult electronic properties: {e_el}"
        )


def postprocess_orca_qm_result(
    orca_run: OrcaRun,
    qm_result: QMResult,
    node_runner: NodeRunner,
    qm_input: Optional[QMInput] = None,
) -> Optional[QMResult_elprop]:
    """Enrich a :class:`QMResult` with additional data parsed from ORCA.

    This helper mirrors the "parsing tail" of the monolithic :func:`orca`
    and :func:`orca_jinja` nodes so that piecewise workflows can yield
    identical results. It performs the following steps:

    * Logs basic electronic properties (dipole, hyperpolarizability).
    * Optionally assigns SMILES / formula onto the final and intermediate
      structures if ``qm_input`` is provided.
    * Parses excited states, absorption spectra, Mayer bond analysis and
      vibrational data from ``orca.out``.
    * Constructs and returns a :class:`QMResult_elprop` instance derived
      directly from :class:`OrcaRun`.
    """

    _log_elprop_basic(orca_run, node_runner)

    # Attach SMILES / formula information where possible.
    if qm_input is not None and getattr(qm_input, "molecule", None) is not None:
        try:
            if qm_result.final_structure:
                qm_result.final_structure.smiles = qm_input.molecule.smiles
                qm_result.final_structure.formula = qm_input.molecule.formula
            if qm_result.structures and getattr(qm_result.structures, "molecules", None):
                for molecule in qm_result.structures.molecules:
                    molecule.smiles = qm_input.molecule.smiles
                    molecule.formula = qm_input.molecule.formula
        except Exception as e_smiles:  # pragma: no cover - defensive
            node_runner.warning(f"Failed to propagate SMILES/formula onto QMResult: {e_smiles}")

    # Parse additional information from orca.out, if available.
    try:
        with open("orca.out") as f:
            contents = f.read()
    except Exception as e_read:  # pragma: no cover - defensive
        node_runner.warning(f"Failed to read orca.out for postprocessing: {e_read}")
        contents = ""

    if contents:
        # Excited states and transitions
        try:
            states_table, transition_table = parse_orca_excited_states(contents)
            if states_table and len(states_table.row) > 0:
                qm_result.excited_states = states_table
            if transition_table and len(transition_table.row) > 0:
                qm_result.excited_state_transitions = transition_table
            node_runner.info("done ORCA excited states parsing")
        except Exception as e:
            node_runner.error(f"Error parsing ORCA excited states: {str(e)}")

        # Absorption spectrum
        try:
            absorption_spectrum = parse_orca_absorption_spectrum(contents)
            if absorption_spectrum and len(absorption_spectrum.row) > 0:
                qm_result.absorption_spectrum = absorption_spectrum
            node_runner.info("done ORCA absorption spectrum parsing")
        except Exception as e:
            node_runner.error(f"Error parsing ORCA absorption spectrum: {str(e)}")

        # Mayer analysis
        try:
            mayer_analysis, mayer_bond_orders = parse_mayer_analysis(contents)
            if mayer_analysis and len(mayer_analysis.row) > 0:
                qm_result.mayer_analysis = mayer_analysis
            if mayer_bond_orders and len(mayer_bond_orders.row) > 0:
                qm_result.mayer_bond_orders = mayer_bond_orders
            node_runner.info("done ORCA mayer analysis parsing")
        except Exception as e:
            node_runner.error(f"Error parsing ORCA mayer analysis: {str(e)}")

        # Vibrational data
        try:
            vibrational_frequencies = parse_vibrational_frequencies(contents)
            if vibrational_frequencies and len(vibrational_frequencies.row) > 0:
                qm_result.vibrational_frequencies = vibrational_frequencies

            normal_modes = parse_normal_modes(contents)
            if normal_modes and len(normal_modes.row) > 0:
                qm_result.normal_modes = normal_modes

            ir_spectrum = parse_ir_spectrum(contents)
            if ir_spectrum and len(ir_spectrum.row) > 0:
                qm_result.ir_spectrum = ir_spectrum
            node_runner.info("done ORCA vibrational frequencies parsing")
        except Exception as e:
            node_runner.error(f"Error parsing ORCA vibrational frequencies: {str(e)}")

    # Build and return the dedicated electronic-properties result.
    try:
        elprop_result = QMResult_elprop.from_orca_output(
            orca_run,
            parent_qm_result=qm_result,
            task_id=node_runner.task_id,
        )
        node_runner.info("QMResult_elprop created from OrcaRun (postprocess helper)")
        return elprop_result
    except Exception as e_elprop:  # pragma: no cover - defensive
        node_runner.warning(f"Failed to construct QMResult_elprop from OrcaRun: {e_elprop}")
        return None


def _normalise_run_result(run_result: object) -> Iterable[FileStack]:
    """Normalise various payload shapes returned from ``orca_run_only``.

    Nested node calls within Simstack may yield bare lists of FileStacks,
    FileListModel / FileListIO wrappers, or objects with a ``files``
    attribute. This helper converts these into a simple iterable of
    :class:`FileStack` instances.
    """

    if isinstance(run_result, list):
        return run_result
    if isinstance(run_result, FileListModel):
        return list(run_result.file_stacks)
    if isinstance(run_result, FileListIO):
        return list(run_result.file_list.file_stacks)
    if hasattr(run_result, "files"):
        return list(getattr(run_result, "files"))
    raise TypeError(f"Unsupported run_result type {type(run_result)}")


@node(parameters=parameters)
async def orca_input_file_only(qm_input: QMInput, **kwargs) -> SimstackResult:
    """Generate only the ORCA input file and return it as a FileStack in a FileListIO.

    This node reproduces the *input generation* part of :func:`orca` but does
    not run ORCA. The resulting input file (``orca.inp``) is packaged into a
    :class:`FileListIO` so it can be passed to subsequent workflow nodes.

    SimstackResult:
        file_list_io (FileListIO): File list containing the generated
            ``orca.inp`` file (as a single FileStack).
    """

    from .orca_main_lib import (
        add_grid_to_simple_input_line,
        add_tddft_block_if_needed,
        add_electronic_properties_block,
        make_casscf_block,
        use_gbw_restart,
    )

    from molecular_qm_models import AuxBasisEnum

    node_runner = NodeRunner("orca_input_file_only", logger=logger, **kwargs)
    try:
        parameters_local = kwargs.get("parameters", None)
        try:
            # Use this node's own parameters object instead of the monolithic
            # ``orca`` node to avoid cross-module coupling.
            slurm_params = orca_input_file_only._node_parameters.slurm_parameters
            node_runner.info(
                f"Slurm parameters: time={slurm_params.time}, mem={slurm_params.mem}, "
                f"cpus_per_task={slurm_params.cpus_per_task}, tasks-per-node={slurm_params.tasks_per_node}"
            )
            pal_block = ""
            mem_block = ""
            if (
                slurm_params.cpus_per_task is not None
                and slurm_params.cpus_per_task > 0
                and slurm_params.tasks_per_node is not None
                and slurm_params.tasks_per_node > 0
            ):
                pal_block = (
                    f"%pal\n nprocs {int(slurm_params.cpus_per_task) * int(slurm_params.tasks_per_node)}\nend\n\n"
                )
                n_cpus_eff = int(slurm_params.cpus_per_task) * int(
                    slurm_params.tasks_per_node
                )
            else:
                node_runner.warning(
                    "Slurm parameter 'cpus_per_task' is not set or invalid, defaulting to 1 CPU."
                )
                n_cpus_eff = 1

            if slurm_params.mem is not None and slurm_params.mem != "":
                if "G" in slurm_params.mem:
                    total_memory = int(slurm_params.mem.replace("G", "").strip()) * 1024
                    memory_per_cpu = int(float(total_memory) / n_cpus_eff) + 100
                    mem_block = f"%maxcore {memory_per_cpu}\n"
                elif "M" in slurm_params.mem:
                    memory_per_cpu = int(
                        float(slurm_params.mem.replace("M", "").strip()) / n_cpus_eff
                    ) + 100
                    mem_block = f"%maxcore {memory_per_cpu}\n"
                else:
                    node_runner.warning(
                        "Slurm parameter 'mem' is not in expected format (e.g., '2G'), defaulting to 2000 MB total memory."
                    )
                    mem_block = "%maxcore 2000\n"
        except Exception as e:
            node_runner.warning(
                f"Could not retrieve Slurm parameters within orca_input_file_only function: {str(e)}"
            )
            pal_block = ""
            mem_block = ""

        if parameters_local:
            node_runner.info(f"Using parameters: {parameters_local}")

        node_runner.info(
            f"multiplicity: {qm_input.multiplicity} optimize: {qm_input.optimization}"
        )
        node_runner.info(f"charge: {qm_input.charge} states: {qm_input.states}")
        node_runner.info(
            f"active_electrons: {qm_input.active_electrons} active_orbitals: {qm_input.active_orbitals}"
        )
        node_runner.info(f"solvent: {qm_input.solvent}")

        blocks = []
        if pal_block:
            blocks.append(pal_block)
        if mem_block:
            blocks.append(mem_block)

        if qm_input.active_electrons > 0:  # casscf
            first_line, cas_scf_spec_bloc, cas_scf_output_block = make_casscf_block(
                qm_input
            )
            blocks.append(cas_scf_spec_bloc)
            blocks.append(cas_scf_output_block)
        else:
            aux_basis = (
                qm_input.basis_set.aux_basis.aux_basis.value
                if qm_input.basis_set.aux_basis.aux_basis != AuxBasisEnum.NONE
                else ""
            )

            first_line = (
                f"{qm_input.functional.functional.value} {qm_input.basis_set.basis_set.value} "
                + aux_basis
            )

            if qm_input.multiplicity > 1 or qm_input.open_shell_calculation:
                first_line += " UHF"
            if qm_input.optimization:
                first_line += " Opt"
                geom_block = (
                    f"%geom\n MaxIter {qm_input.max_optimization_iterations}\nend\n\n"
                )
                blocks.append(geom_block)
            elif qm_input.gradients:
                first_line += " ENGRAD"

            node_runner.info(f"frequencies: {qm_input.frequencies}")
            if qm_input.frequencies:
                first_line += " FREQ"

            if qm_input.solvent.lower() != "none":
                first_line += f" CPCM({qm_input.solvent.upper()})"

            first_line = add_grid_to_simple_input_line(first_line, qm_input)

            blocks = add_electronic_properties_block(qm_input, blocks)
            conv_keyword = qm_input.scf_accuracy.value
            scf_block = (
                f"%scf\n Convergence {conv_keyword}\n MaxIter {qm_input.max_scf_iterations}\nend\n\n"
            )
            blocks.append(scf_block)

            blocks = add_tddft_block_if_needed(qm_input, blocks)

        constraints = qm_input.molecule.properties.get("constraints", [])
        node_runner.info(f"constraints: {constraints}")

        frozen_atoms = [
            c["index"] + 1
            for c in constraints
            if c["type"] == "frozen" and "index" in c
        ]

        if frozen_atoms:
            frozen_atoms_block = "%geom\n Constraints\n"
            for atom_idx in frozen_atoms:
                frozen_atoms_block += f"  {{ C {atom_idx} C }}\n"

            frozen_atoms_block += "end\nend\n\n"
            blocks.append(frozen_atoms_block)

        with_geometry = True
        if os.path.exists("orca.xyz"):
            node_runner.info("found orca.xyz file. Using it for geometry optimization. ")
            with_geometry = False
            blocks.append(f"* xyzfile {qm_input.charge} {qm_input.multiplicity} orca.xyz\n")
        if os.path.exists("orca.gbw"):
            node_runner.info("found orca.gbw file. Using it for geometry optimization. ")
            first_line += " MORead"
            blocks.append("%moinp \"orca.gbw\"\n\n")
        else:
            if qm_input.restart_files is not None and len(qm_input.restart_files) > 0:
                first_line, blocks = use_gbw_restart(
                    node_runner, qm_input, first_line, blocks
                )

        if qm_input.first_line is not None and qm_input.first_line != "":
            first_line += f" {qm_input.first_line}"
        first_line += "\n\n"
        params = [first_line]

        orca_input_file = OrcaInput(
            qm_input.molecule,
            charge=qm_input.charge,
            spin_multiplicity=qm_input.multiplicity,
            input_parameters=params,
            blocks=blocks,
            with_geometry=with_geometry,
        )

        with open("orca.inp", "w") as f:
            f.write(str(orca_input_file))
            node_runner.info(f"orca.inp contents:\n{str(orca_input_file)}")

        input_stack = FileStack.from_local_file(
            "orca.inp", in_memory=True, is_hashable=True, secure_source=True
        )
        node_runner.info_files.append(input_stack)

        file_list_io = FileListIO()
        file_list_io.file_list.append(input_stack)
        node_runner.file_list_io = file_list_io

        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error generating ORCA input file: {str(e)}")


@node(parameters=parameters)
async def orca_run_only(file_list_io: FileListIO, **kwargs) -> SimstackResult:
    """Run ORCA given an existing ``orca.inp`` (passed via FileListIO) and
    return the produced output files as FileStacks.

    Assumes that the first entry in ``file_list_io.file_list`` is the
    ``orca.inp`` FileStack.

    SimstackResult:
        files (List[FileStack]): Collected ORCA output and log files.
    """

    node_runner = NodeRunner("orca_run_only", logger=logger, **kwargs)
    try:
        if not file_list_io.file_list.file_stacks:
            return node_runner.fail("No input files provided in FileListIO")

        # Retrieve orca.inp into current working directory
        input_stack = file_list_io.file_list.file_stacks[0]
        local_path = input_stack.get(local_dir=Path(""))
        if local_path.name != "orca.inp":
            # Ensure expected name
            os.replace(local_path, "orca.inp")

        # parent_parameters.resource is a Resource object; convert to its string
        # value before doing membership tests to avoid "unhashable type: 'Resource'".
        resource_obj = kwargs["parent_parameters"].resource
        resource = getattr(resource_obj, "value", str(resource_obj))

        if resource in {"local", "self", "local-home"}:
            try:
                shell_check = subprocess.run(
                    ["/bin/sh", "-c", "echo $SHELL"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                shell = (shell_check.stdout or "").strip()
            except Exception:
                shell = os.environ.get("SHELL", "")

            locator_cmd = "which" if shell and "bash" in shell else "where"
            command = f"{locator_cmd} orca && orca orca.inp > orca.out"
        elif resource == "int-nano":
            command = (
                "source ~/.bashrc && module load orca/6.1.1 && $(which orca) orca.inp > orca.out"
            )
        else:
            raise RuntimeError(f"Unsupported resource {resource}")

        if not node_runner.subprocess("orca_run", command):
            if os.path.exists("orca.out"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca.out",
                        in_memory=True,
                        is_hashable=True,
                        secure_source=True,
                    )
                )
            if os.path.exists("orca_run.log"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca_run.log",
                        in_memory=True,
                        is_hashable=True,
                        secure_source=True,
                    )
                )
            return node_runner.fail("Error running ORCA calculation")

        # Collect main output files as FileStacks
        output_files = []
        for fname in [
            "orca.out",
            "orca.engrad",
            "orca.opt",
            "orca.property.txt",
            "orca.xyz",
            "orca.gbw",
            "orca.densities",
            "orca.hess",
            "orca_trj.xyz",
            "orca.DFTMRCI.inp",
            "orca.bkji",
        ]:
            if os.path.exists(fname):
                output_files.append(
                    FileStack.from_local_file(
                        fname,
                        in_memory=True,
                        is_hashable=True,
                        secure_source=True,
                    )
                )
        node_runner.files.extend(output_files)

        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error running ORCA: {str(e)}")


@node(parameters=parameters)
async def orca_collect_results(file_list_io: FileListIO, **kwargs) -> SimstackResult:
    """Collect an existing ORCA output (``orca.out`` and related files) and
    construct both :class:`QMResult` and :class:`QMResult_elprop`.

    Assumes that ``file_list_io.file_list`` contains at least the
    ``orca.out`` file, and optionally other ORCA output files used by
    :meth:`QMResult.from_orca_output`.

    SimstackResult:
        orca_result (QMResult): Parsed result from the ORCA calculations.
        orca_elprop_result (QMResult_elprop, optional): Electronic
            properties result derived from the same ORCA output.
    """

    # ``qm_input`` is optional and may be provided by combined workflows
    # (``orca_combined`` / ``orca_jinja_combined``) so that SMILES/formula
    # information can be propagated onto the resulting structures, matching
    # the behaviour of the monolithic :func:`orca` / :func:`orca_jinja`.
    qm_input: Optional[QMInput] = kwargs.get("qm_input", None)

    node_runner = NodeRunner("orca_collect_results", logger=logger, **kwargs)
    try:
        if not file_list_io.file_list.file_stacks:
            return node_runner.fail("No files provided in FileListIO")

        local_dir = Path("")
        # Materialize all provided files into current working directory
        for fs in file_list_io.file_list.file_stacks:
            fs.get(local_dir=local_dir)

        if not os.path.exists("orca.out"):
            return node_runner.fail("orca.out file not found")

        node_runner.info_files.append(
            FileStack.from_local_file(
                "orca.out", in_memory=True, is_hashable=True, secure_source=True
            )
        )
        orca_run = OrcaRun("orca")

        qm_result = await QMResult.from_orca_output(orca_run, node_runner.task_id)

        # Enrich the QMResult with the same post-processing used by the
        # monolithic ORCA nodes (excited states, spectra, vibrations, etc.)
        # and derive the dedicated elprop result. ``qm_input`` is optional
        # and, when provided, is only used to attach SMILES/formula data.
        elprop_result = postprocess_orca_qm_result(orca_run, qm_result, node_runner, qm_input=qm_input)
        if elprop_result is not None:
            node_runner.orca_elprop_result = elprop_result
        node_runner.orca_result = qm_result

        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error collecting ORCA results: {str(e)}")


@node(parameters=parameters)
async def orca_combined(qm_input: QMInput, **kwargs) -> SimstackResult:
    """Combined workflow that chains ``orca_input_file_only``,
    ``orca_run_only`` and ``orca_collect_results`` to reproduce the
    behavior of the monolithic :func:`orca` node.

    This node exists mainly for workflow composition via FileStack / FileListIO.
    """

    node_runner = NodeRunner("orca_combined", logger=logger, **kwargs)
    try:
        # Step 1: generate input file
        try:
            file_list_io = await orca_input_file_only(qm_input, **kwargs)
        except Exception as e:
            return node_runner.fail(f"orca_input_file_only failed: {e}")

        if not isinstance(file_list_io, FileListIO):
            node_runner.warning(
                f"orca_input_file_only returned unexpected type {type(file_list_io)}; "
                "expected FileListIO."
            )

        # Step 2: run ORCA with the generated input
        try:
            run_result = await orca_run_only(file_list_io, **kwargs)
        except Exception as e:
            return node_runner.fail(f"orca_run_only failed: {e}")

        try:
            run_files = list(_normalise_run_result(run_result))
        except TypeError as e:
            return node_runner.fail(str(e))

        # Build FileListIO from produced files for the collector node
        run_files_io = FileListIO()
        for fs in run_files:
            run_files_io.file_list.append(fs)

        # Step 3: collect results into QMResult (+ QMResult_elprop). Pass
        # ``qm_input`` so that SMILES/formula information can be attached
        # to the resulting structures, mirroring the monolithic ``orca``
        # node behaviour.
        try:
            collect_result = await orca_collect_results(run_files_io, qm_input=qm_input, **kwargs)
        except Exception as e:
            return node_runner.fail(f"orca_collect_results failed: {e}")

        # In nested usage, orca_collect_results is expected to return the
        # QMResult directly, but we also support the case where it returns
        # an object with an "orca_result" attribute.
        if hasattr(collect_result, "orca_result"):
            orca_result = collect_result.orca_result
        else:
            orca_result = collect_result

        node_runner.orca_result = orca_result

        # Propagate the elprop result if it was already constructed by the
        # collector; otherwise fall back to deriving it from the QMResult for
        # backwards compatibility.
        elprop_result = None
        if hasattr(collect_result, "orca_elprop_result"):
            elprop_result = getattr(collect_result, "orca_elprop_result")
        if elprop_result is None:
            try:
                elprop_result = QMResult_elprop.from_qm_result(orca_result)
                node_runner.info(
                    "QMResult_elprop created from QMResult (fallback in orca_combined node)"
                )
            except Exception as e_elprop:  # pragma: no cover - defensive
                node_runner.warning(
                    f"Failed to construct QMResult_elprop from QMResult in orca_combined: {e_elprop}"
                )

        if elprop_result is not None:
            node_runner.orca_elprop_result = elprop_result

        node_runner.files.extend(run_files)

        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error in orca_combined workflow: {str(e)}")


@node(parameters=parameters)
async def orca_jinja_run_only(file_list_io: FileListIO, **kwargs) -> SimstackResult:
    """Run ORCA using the configuration-driven command from ``config.toml``.

    This node mirrors :func:`orca_run_only` but constructs the execution
    command via :func:`_build_orca_command_from_config`, with transparent
    fallback to the same hard-coded commands used by the legacy node.
    """

    node_runner = NodeRunner("orca_jinja_run_only", logger=logger, **kwargs)
    try:
        if not file_list_io.file_list.file_stacks:
            return node_runner.fail("No input files provided in FileListIO")

        input_stack = file_list_io.file_list.file_stacks[0]
        local_path = input_stack.get(local_dir=Path(""))
        if local_path.name != "orca.inp":
            os.replace(local_path, "orca.inp")

        resource_obj = kwargs["parent_parameters"].resource
        resource = getattr(resource_obj, "value", str(resource_obj))

        # Prefer config.toml based command; fall back to legacy behaviour.
        command = _build_orca_command_from_config(resource, "orca")
        if command is None:
            if resource in {"local", "self", "local-home"}:
                try:
                    shell_check = subprocess.run(
                        ["/bin/sh", "-c", "echo $SHELL"],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    shell = (shell_check.stdout or "").strip()
                except Exception:
                    shell = os.environ.get("SHELL", "")

                locator_cmd = "which" if shell and "bash" in shell else "where"
                command = f"{locator_cmd} orca && orca orca.inp > orca.out"
            elif resource == "int-nano":
                command = "source ~/.bashrc && module load orca/6.1.1 && $(which orca) orca.inp > orca.out"
            else:
                raise RuntimeError(f"Unsupported resource {resource}")

        if not node_runner.subprocess("orca_run", command):
            if os.path.exists("orca.out"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca.out",
                        in_memory=True,
                        is_hashable=True,
                        secure_source=True,
                    )
                )
            if os.path.exists("orca_run.log"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca_run.log",
                        in_memory=True,
                        is_hashable=True,
                        secure_source=True,
                    )
                )
            return node_runner.fail("Error running ORCA calculation")

        output_files = []
        for fname in [
            "orca.out",
            "orca.engrad",
            "orca.opt",
            "orca.property.txt",
            "orca.xyz",
            "orca.gbw",
            "orca.densities",
            "orca.hess",
            "orca_trj.xyz",
            "orca.DFTMRCI.inp",
            "orca.bkji",
        ]:
            if os.path.exists(fname):
                output_files.append(
                    FileStack.from_local_file(
                        fname,
                        in_memory=True,
                        is_hashable=True,
                        secure_source=True,
                    )
                )
        node_runner.files.extend(output_files)

        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error running ORCA (jinja): {str(e)}")


@node(parameters=parameters)
async def orca_jinja_combined(qm_input: QMInput, **kwargs) -> SimstackResult:
    """Combined workflow that chains ``orca_input_file_only``,
    ``orca_jinja_run_only`` and ``orca_collect_results`` to reproduce the
    behavior of the monolithic :func:`orca_jinja` node.

    This is the configuration-driven analogue of :func:`orca_combined`.
    """

    node_runner = NodeRunner("orca_jinja_combined", logger=logger, **kwargs)
    try:
        # Step 1: generate input file
        try:
            file_list_io = await orca_input_file_only(qm_input, **kwargs)
        except Exception as e:
            return node_runner.fail(f"orca_input_file_only failed: {e}")

        if not isinstance(file_list_io, FileListIO):
            node_runner.warning(
                f"orca_input_file_only returned unexpected type {type(file_list_io)}; "
                "expected FileListIO."
            )

        # Step 2: run ORCA with a config-driven command
        try:
            run_result = await orca_jinja_run_only(file_list_io, **kwargs)
        except Exception as e:
            return node_runner.fail(f"orca_jinja_run_only failed: {e}")

        try:
            run_files = list(_normalise_run_result(run_result))
        except TypeError as e:
            return node_runner.fail(str(e))

        run_files_io = FileListIO()
        for fs in run_files:
            run_files_io.file_list.append(fs)

        # Step 3: collect results into QMResult (+ QMResult_elprop). As in
        # ``orca_combined``, we pass ``qm_input`` so that structural
        # metadata can be populated consistently with the monolithic
        # ``orca_jinja`` node.
        try:
            collect_result = await orca_collect_results(run_files_io, qm_input=qm_input, **kwargs)
        except Exception as e:
            return node_runner.fail(f"orca_collect_results failed: {e}")

        if hasattr(collect_result, "orca_result"):
            orca_result = collect_result.orca_result
        else:
            orca_result = collect_result

        node_runner.orca_result = orca_result

        elprop_result = None
        if hasattr(collect_result, "orca_elprop_result"):
            elprop_result = getattr(collect_result, "orca_elprop_result")
        if elprop_result is None:
            try:
                elprop_result = QMResult_elprop.from_qm_result(orca_result)
                node_runner.info(
                    "QMResult_elprop created from QMResult (fallback in orca_jinja_combined node)"
                )
            except Exception as e_elprop:  # pragma: no cover - defensive
                node_runner.warning(
                    f"Failed to construct QMResult_elprop from QMResult in orca_jinja_combined: {e_elprop}"
                )

        if elprop_result is not None:
            node_runner.orca_elprop_result = elprop_result

        node_runner.files.extend(run_files)

        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error in orca_jinja_combined workflow: {str(e)}")
