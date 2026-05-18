"""ORCA restart nodes: start new ORCA jobs from existing results.

This module implements the functionality described in the master task:

* Analyse completed ORCA jobs (classic or jinja-based) via their
  :class:`NodeRegistry` entries.
* Extract the corresponding :class:`QMInput` and :class:`QMResult` from
  the database.
* Build a new :class:`QMInput` that reuses
  - the *final geometry* from the previous QMResult and
  - all *electronic/QM settings* (functional, basis set, dispersion,
    charge, multiplicity, method, states, etc.) from the original
    QMInput,
  while overriding
  - optimisation/frequency flags and
  - accuracy related options.
* Run a new ORCA calculation with tightened settings:

    VERYTIGHTOPT DEFGRID3 VeryTightSCF  FREQ

  (no duplicate plain ``Opt`` keyword on the first line).

Two restart nodes are provided:

* :func:`orca_start_from_result` – uses the same hard-coded ORCA command
  logic as the classic :func:`orca` node.
* :func:`orca_start_from_result_jinja` – uses
  :func:`_build_orca_command_from_config` from :mod:`orca` to construct
  the ORCA command from ``config.toml`` (jinja/config driven).

Additionally a small master node

* :func:`many_orca_restart_from_results`

is provided which accepts a :class:`StringList` of previously completed
ORCA NodeRegistry ids and runs the restart node for each of them.

Existing nodes in :mod:`applications.electronic_structure.orca.orca`
remain untouched.
"""

import asyncio
import logging
import os
import subprocess
from typing import Tuple

from odmantic import ObjectId

from simstack.core.context import context
from simstack.core.definitions import TaskStatus
from simstack.core.engine import current_engine_context
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner

from simstack.models import NodeRegistry, StringData
from simstack.models.files import FileStack

from molecular_qm_models import (
    GridType,
    QMInput,
    SCFAccuracy,
)
from molecular_qm_models import AuxBasisEnum
from molecular_qm_models.qm_result import QMResult
from molecular_qm_models import Molecule
from molecular_qm_models import (
    _build_orca_command_from_config,
    parameters as orca_parameters,
)
from molecular_qm_models import OrcaInput, OrcaRun
from molecular_qm_models import (
    add_electronic_properties_block,
    add_tddft_block_if_needed,
    set_method_and_basis_set_for_non_casscf_methods,
    set_orca_memory_and_pal_options_according_to_slurm_parameters,
    use_gbw_restart,
)
from simstack.models.base_lists import StringList

from molecular_qm_orca.orca import (
    load_node_inputs,
    load_node_outputs,
)


logger = logging.getLogger("OrcaRestartNode")


# ---------------------------------------------------------------------------
# Helper: resolve QMInput/QMResult for a given NodeRegistry id
# ---------------------------------------------------------------------------


async def _resolve_qminput_qmresult_from_registry_id(
    registry_id: str,
) -> Tuple[QMInput, QMResult]:
    """Resolve (QMInput, QMResult) for an existing ORCA NodeRegistry entry.

    Parameters
    ----------
    registry_id
        Hex string representation of the :class:`NodeRegistry.id` for a
        completed ``orca`` or ``orca_jinja`` node.
    """

    if not context.initialized:
        await context.initialize(path=__file__)

    engine = current_engine_context.get()

    try:
        oid = ObjectId(registry_id)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid NodeRegistry id '{registry_id}': {exc}") from exc

    registry_entry = await engine.find_one(NodeRegistry, NodeRegistry.id == oid)
    if registry_entry is None:
        raise RuntimeError(f"No NodeRegistry entry found for id={registry_id}")

    status = getattr(registry_entry, "status", None)
    if status is not TaskStatus.COMPLETED:
        raise RuntimeError(
            f"NodeRegistry entry {registry_id} has non-completed status: {status}"
        )

    node_inputs = await load_node_inputs(registry_entry)
    node_outputs = await load_node_outputs(registry_entry)
    if not node_inputs or not node_outputs:
        raise RuntimeError(
            f"Could not resolve inputs/outputs for NodeRegistry {registry_id}"
        )

    # Locate QMInput
    original_qm_input: QMInput | None = None
    candidate = node_inputs.get("QMInput")
    if isinstance(candidate, QMInput):
        original_qm_input = candidate
    else:
        for obj in node_inputs.values():
            if isinstance(obj, QMInput):
                original_qm_input = obj
                break

    # Locate QMResult
    original_qm_result: QMResult | None = None
    candidate = node_outputs.get("orca_result")
    if isinstance(candidate, QMResult):
        original_qm_result = candidate
    else:
        for obj in node_outputs.values():
            if isinstance(obj, QMResult):
                original_qm_result = obj
                break

    if original_qm_input is None or original_qm_result is None:
        raise RuntimeError(
            f"Could not resolve QMInput/QMResult for NodeRegistry {registry_id}"
        )

    return original_qm_input, original_qm_result


def _extract_final_structure(qm_result: QMResult) -> Molecule:
    """Return the final structure from a QMResult.

    Preference order:

    1. ``qm_result.final_structure`` if present.
    2. Last entry in ``qm_result.structures.molecules``.
    """

    if qm_result.final_structure is not None:
        return qm_result.final_structure

    structures = getattr(qm_result, "structures", None)
    if structures is not None:
        mols = getattr(structures, "molecules", [])
        if mols:
            return mols[-1]

    raise RuntimeError("QMResult does not contain a usable final structure")


def build_restart_qm_input(original: QMInput, qm_result: QMResult) -> QMInput:
    """Create a new QMInput for a restart job from previous input/result.

    The new QMInput:

    * uses the *final structure* from ``qm_result`` as the geometry;
    * keeps charge, multiplicity, method, basis set, functional,
      dispersion, states, etc., from ``original``;
    * enables geometry optimisation and frequency calculation;
    * tightens convergence / grid settings and injects

        VERYTIGHTOPT DEFGRID3 VeryTightSCF

      via the ``first_line`` field, without relying on the plain
      ``Opt`` keyword.
    """

    final_structure = _extract_final_structure(qm_result)

    # Deep copy to preserve the original QMInput document.
    restart = original.model_copy(deep=True)

    # Geometry from the result; keep everything else as configured.
    restart.molecule = final_structure

    # Ensure optimisation + frequencies are enabled.
    restart.optimization = True
    restart.frequencies = True

    # Tighten SCF and grid settings; rely on the restart node to *not*
    # add another plain "Opt" but to use VERYTIGHTOPT instead.
    restart.scf_accuracy = SCFAccuracy.VeryTight
    restart.grid_type = GridType.Grid3

    # Explicit first-line accuracy / grid keywords. The restart-specific
    # ORCA node will construct a first line that contains the method,
    # basis set and these options, *without* appending a duplicate "Opt".
    restart.first_line = "VERYTIGHTOPT DEFGRID3 VeryTightSCF"

    # Make sure gradients do not conflict with optimisation/frequencies.
    restart.gradients = False

    return restart


# ---------------------------------------------------------------------------
# Core ORCA restart runner (shared by classic and jinja variants)
# ---------------------------------------------------------------------------


async def _run_restart_orca(
    node_runner: NodeRunner,
    qm_input: QMInput,
    use_jinja_command: bool,
    **kwargs,
):
    """Generate ORCA input for a restart and run the calculation.

    This mirrors the structure of :func:`orca` / :func:`orca_jinja` but
    customises the first-line construction so that we can enforce

        VERYTIGHTOPT DEFGRID3 VeryTightSCF FREQ

    without ever emitting a plain ``Opt`` keyword.
    """

    try:
        # Slurm / memory and PAL options from helper (same as orca/orca_jinja)
        blocks = set_orca_memory_and_pal_options_according_to_slurm_parameters(
            node_runner,
            blocks=None,
            kwargs=kwargs,
            node_slurm_params=orca_parameters.slurm_parameters,
        )

        # We currently support only non-CASSCF restart jobs. If an
        # active-space calculation is encountered, fall back to the
        # generic method/basis first-line generator but still apply the
        # tightened accuracy settings.
        aux_field = qm_input.basis_set.aux_basis.aux_basis
        # Mirror the behaviour of the main ORCA node: only emit an
        # auxiliary basis label if it is *not* the explicit "NONE"
        # sentinel. Otherwise we would end up with a literal "none" in
        # the simple input line, which ORCA does not accept.
        if aux_field is not None and aux_field != AuxBasisEnum.NONE:
            aux_basis = aux_field.value
        else:
            aux_basis = ""
        first_line = set_method_and_basis_set_for_non_casscf_methods(
            node_runner,
            qm_input,
            aux_basis,
            first_line="",
        )

        if qm_input.multiplicity > 1 or qm_input.open_shell_calculation:
            first_line += " UHF"

        # Inject tightened optimisation/grid/SCF settings and a
        # frequency calculation explicitly. We rely on the caller having
        # set qm_input.first_line to the desired accuracy string but we
        # still guard against duplicates.
        accuracy_segment = (qm_input.first_line or "").strip()
        # Ensure the three desired keywords are present exactly once.
        for token in ("VERYTIGHTOPT", "DEFGRID3", "VeryTightSCF"):
            if token not in accuracy_segment.split():
                accuracy_segment = (accuracy_segment + " " + token).strip()

        # Add frequency keyword once.
        frequency_token = "FREQ" if qm_input.frequencies else ""

        tokens = [
            t
            for t in (accuracy_segment, frequency_token)
            if t is not None and t != ""
        ]
        if tokens:
            first_line += " " + " ".join(tokens)

        # Solvent handling as in the main ORCA node.
        if qm_input.solvent.lower() != "none":
            first_line += f" CPCM({qm_input.solvent.upper()})"

        # Geometry optimisation control block (but no extra "Opt" on
        # the first line).
        if qm_input.optimization:
            geom_block = (
                f"%geom\n MaxIter {qm_input.max_optimization_iterations}\nend\n\n"
            )
            blocks.append(geom_block)

        # Electrical properties, SCF block and TDDFT block as usual.
        blocks = add_electronic_properties_block(qm_input, blocks)

        conv_keyword = qm_input.scf_accuracy.value
        scf_block = (
            f"%scf\n Convergence {conv_keyword}\n MaxIter {qm_input.max_scf_iterations}\nend\n\n"
        )
        blocks.append(scf_block)

        blocks = add_tddft_block_if_needed(qm_input, blocks)

        # Constraints (if any) and restart files (gbw) handling mirror
        # the logic from the main ORCA node.
        constraints = qm_input.molecule.properties.get("constraints", [])
        node_runner.info(f"constraints: {constraints}")

        frozen_atoms = [
            c["index"] + 1
            for c in constraints
            if c.get("type") == "frozen" and "index" in c
        ]
        if frozen_atoms:
            frozen_atoms_block = "%geom\n Constraints\n"
            for atom_idx in frozen_atoms:
                frozen_atoms_block += f"  {{ C {atom_idx} C }}\n"
            frozen_atoms_block += "end\nend\n\n"
            blocks.append(frozen_atoms_block)

        with_geometry = True
        if os.path.exists("orca.xyz"):
            node_runner.info(
                "found orca.xyz file. Using it for geometry optimisation restart. "
            )
            with_geometry = False
            blocks.append(f"* xyzfile {qm_input.charge} {qm_input.multiplicity} orca.xyz\n")

        if os.path.exists("orca.gbw"):
            node_runner.info("found orca.gbw file. Using it for restart.")
            first_line += " MORead"
            blocks.append("%moinp \"orca.gbw\"\n\n")
        else:
            if qm_input.restart_files is not None and len(qm_input.restart_files) > 0:
                first_line, blocks = use_gbw_restart(
                    node_runner, qm_input, first_line, blocks
                )

        # Finalise first line including any user-provided extras (already
        # contained in qm_input.first_line) and terminate with blank line.
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
            node_runner.info(f"orca.inp contents (restart):\n{str(orca_input_file)}")

        node_runner.info_files.append(
            FileStack.from_local_file(
                "orca.inp", in_memory=True, is_hashable=True, secure_source=True
            )
        )
        node_runner.info("restart input files done")
    except Exception as exc:
        return node_runner.fail(f"Error generating restart ORCA input files: {exc}")

    # ------------------------------------------------------------------
    # Run ORCA and parse results (mirrors orca / orca_jinja structure)
    # ------------------------------------------------------------------

    # Determine resource string
    parent_params = kwargs.get("parent_parameters")
    if parent_params is None:
        raise RuntimeError("parent_parameters missing in restart node kwargs")

    resource_obj = parent_params.resource
    resource = getattr(resource_obj, "value", str(resource_obj))

    if use_jinja_command:
        command = _build_orca_command_from_config(resource, "orca")
        if command is None:
            # Fall back to the same defaults as the classic ORCA node
            use_jinja_command = False
    if not use_jinja_command:
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
        else:  # pragma: no cover - defensive
            raise RuntimeError(f"Unsupported resource {resource}")

    if not node_runner.subprocess("orca_restart_run", command):
        if not qm_input.tolerate_failure:
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
            return node_runner.fail("Error running ORCA restart calculation")

    orca_run = None
    try:
        if os.path.exists("orca.out"):
            node_runner.info_files.append(
                FileStack.from_local_file(
                    "orca.out", in_memory=True, is_hashable=True, secure_source=True
                )
            )
            path_without_extension = "orca"
            orca_run = OrcaRun(path_without_extension)
            if not orca_run.normal_termination:
                if qm_input.tolerate_failure:
                    with open("orca.out", "r") as f:
                        orca_out_contents = f.read()
                    msg = (
                        "unfortunately, the SCF has not converged. There may be a way out but we have to stop here"
                    )
                    if msg not in orca_out_contents:
                        return node_runner.fail("orca did not terminate normally")
                    else:
                        node_runner.warning(
                            "orca did not terminate normally but it was tolerated. Continuing execution."
                        )
                else:
                    return node_runner.fail("orca did not terminate normally")
            node_runner.info("restart calculation finished")
        else:
            return node_runner.fail("orca.out file not found for restart")
    except Exception as exc:
        return node_runner.fail(f"orca.out parsing error (restart) {exc}")

    try:
        if orca_run is None:
            return node_runner.fail("orca_run is None in restart node")

        orca_result = await QMResult.from_orca_output(orca_run, node_runner.task_id)
        node_runner.orca_result = orca_result

        return node_runner.succeed()
    except Exception as exc:  # pragma: no cover - defensive
        return node_runner.fail(f"Error reading ORCA restart result: {exc}")


# ---------------------------------------------------------------------------
# Public restart nodes
# ---------------------------------------------------------------------------


@node(parameters=orca_parameters)
async def orca_start_from_result(
    previous_task_id: StringData,
    **kwargs,
):
    """Restart ORCA from a previous result using classic command logic.

    Parameters
    ----------
    previous_task_id
        :class:`StringData` whose ``value`` field contains the hex
        :class:`NodeRegistry.id` of a completed ``orca``/``orca_jinja``
        task.
    """

    node_runner = NodeRunner("orca_start_from_result", logger=logger, **kwargs)

    try:
        orig_input, orig_result = await _resolve_qminput_qmresult_from_registry_id(
            previous_task_id.value
        )
    except Exception as exc:
        return node_runner.fail(str(exc))

    restart_qm_input = build_restart_qm_input(orig_input, orig_result)

    # ``node_runner`` is already passed positionally into the helper; the
    # Simstack wrapper may also inject a ``node_runner`` kwarg into
    # ``kwargs``. To avoid "multiple values for argument 'node_runner'"
    # errors we explicitly drop any such entry before forwarding.
    child_kwargs = dict(kwargs)
    child_kwargs.pop("node_runner", None)

    return await _run_restart_orca(
        node_runner,
        restart_qm_input,
        use_jinja_command=False,
        **child_kwargs,
    )


@node(parameters=orca_parameters)
async def orca_start_from_result_jinja(
    previous_task_id: StringData,
    **kwargs,
):
    """Restart ORCA from a previous result using config.toml command.

    This behaves like :func:`orca_start_from_result` but uses
    :func:`_build_orca_command_from_config` (falling back to the classic
    command if the configuration is missing or incomplete).
    """

    node_runner = NodeRunner("orca_start_from_result_jinja", logger=logger, **kwargs)

    try:
        orig_input, orig_result = await _resolve_qminput_qmresult_from_registry_id(
            previous_task_id.value
        )
    except Exception as exc:
        return node_runner.fail(str(exc))

    restart_qm_input = build_restart_qm_input(orig_input, orig_result)

    child_kwargs = dict(kwargs)
    child_kwargs.pop("node_runner", None)

    return await _run_restart_orca(
        node_runner,
        restart_qm_input,
        use_jinja_command=True,
        **child_kwargs,
    )


@node(parameters=orca_parameters)
async def many_orca_restart_from_results(
    previous_task_ids: StringList,
    **kwargs,
):
    """Master node: run restart jobs for a list of completed ORCA tasks.

    The ``previous_task_ids`` input is a :class:`StringList` whose
    ``elements`` are hex :class:`NodeRegistry.id` strings. For testing
    and demonstration purposes this can be restricted to the two example
    ids mentioned in the task description::

        ["69ae98ac559ef7c280d6b479", "69ae988e559ef7c280d6b475"]

    The node runs :func:`orca_start_from_result` once for each id and
    exposes the list of restart results as
    ``node_runner.restart_results``.
    """

    node_runner = NodeRunner("many_orca_restart_from_results", logger=logger, **kwargs)

    # Forward the effective Parameters object into child nodes (same
    # pattern as the many_orca_master_* nodes).
    params = kwargs.get("parameters") or kwargs.get("parent_parameters")
    child_kwargs = dict(kwargs)
    if params is not None:
        child_kwargs["parameters"] = params

    from simstack.core.definitions import TaskStatus as _TS

    # Normalise and filter the list of ids once up front.
    id_strings: list[str] = []
    for task_id_str in previous_task_ids.elements:
        s = (task_id_str or "").strip()
        if s:
            id_strings.append(s)

    if not id_strings:
        node_runner.restart_results = []
        return node_runner.succeed()

    # Spawn one child restart node per id and wait for all of them
    # concurrently. This mirrors how other master nodes fan out work
    # across multiple children while keeping a single logical parent
    # for the GUI/DB.
    tasks = []
    for task_id_str in id_strings:
        id_model = StringData(field_name="previous_task_id", value=task_id_str)
        tasks.append(
            orca_start_from_result(previous_task_id=id_model, **child_kwargs)
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    restart_results = []
    for task_id_str, result in zip(id_strings, results):
        if isinstance(result, Exception):
            return node_runner.fail(
                f"Restart ORCA job for task_id={task_id_str} raised exception: {result}"
            )

        status = getattr(result, "status", None)
        if isinstance(status, _TS):
            is_completed = status is _TS.COMPLETED
        elif isinstance(status, str):
            is_completed = status == "COMPLETED"
        else:
            is_completed = False

        if not is_completed:
            return node_runner.fail(
                f"Restart ORCA job for task_id={task_id_str} failed: "
                f"{getattr(result, 'error_message', None)}"
            )

        restart_results.append((task_id_str, getattr(result, "orca_result", None)))

    node_runner.restart_results = restart_results
    return node_runner.succeed()

