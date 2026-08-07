import os
import subprocess
from typing import List, Optional
from simstack.core.context import context
from simstack.core.node import node
from simstack.core.simstack_result import SimstackResult
from simstack.models.files import FileStack

from applications.electronic_structure import QMInput, QMResult, QMResultElProp
from .lib.orbital_energies_parser import parse_orbital_energies
from .lib.orca_absorption_spectrum_parser import parse_orca_absorption_spectrum
from .lib.orca_excited_states_parser import parse_orca_excited_states
from .lib.orca_frequency_parser import parse_vibrational_frequencies, parse_normal_modes, parse_ir_spectrum
from .lib.orca_mayer_parser import parse_mayer_analysis
from .orca_input import orca_input_factory
import logging
from .orca_output import OrcaOutput

logger = logging.getLogger(__name__)


SCF_NOT_CONVERGED_MSG = (
    "unfortunately, the SCF has not converged. There may be a way out but we have to stop here"
)

ORCA_RESULT_FILES = [
    "orca.out",
    "orca.gbw",
    "orca.xyz",
    "orca.trj",
    "orca.densities",
    "orca.engrad",
    "orca.opt",
    "orca.property",
    "orca_run.log",
]


def orca_run_command(input_files: List[str], result_files: List[str], arg_hash: str):
    try:
        context.resource_config.run("orca", input_files, result_files)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode


def _remove_restart_artifacts():
    """Remove local ORCA restart artifacts so a clean retry cannot pick them up."""
    for path in ("orca.gbw", "orca.xyz", "orca.opt", "orca_traj.xyz"):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Could not remove restart artifact %s", path)


async def _write_orca_input(qm_input: QMInput, use_restart: bool, **kwargs) -> bool:
    """Generate ``orca.inp`` and return whether restart data was incorporated."""
    orca_input_gen = orca_input_factory(qm_input, use_restart=use_restart, **kwargs)

    # Await blocks first so molecular_input_block can update _first_line
    # (e.g. MORead) before first_line is read.
    blocks = await orca_input_gen.blocks()
    first_line = orca_input_gen.first_line

    with open("orca.inp", "w") as f:
        f.write(first_line)
        for block in blocks:
            f.write(block)

    return orca_input_gen.used_restart


def _is_tolerated_abnormal_termination(qm_input: QMInput, orca_out_contents: Optional[str]) -> bool:
    return (
        qm_input.tolerate_failure
        and orca_out_contents is not None
        and SCF_NOT_CONVERGED_MSG in orca_out_contents
    )


async def _retry_orca_without_restart(qm_input: QMInput, reason: str, **kwargs) -> int:
    """Clear restart data, regenerate input, and re-run ORCA."""
    node_runner = kwargs["node_runner"]
    node_runner.warning(f"{reason}; retrying without restart data.")
    _remove_restart_artifacts()
    await _write_orca_input(qm_input, use_restart=False, **kwargs)
    node_runner.info_files.append(
        FileStack.from_local_file("orca.inp", in_memory=True, is_hashable=True, secure_source=True)
    )
    node_runner.info("input files done (retry without restart)")
    return orca_run_command(["orca.inp"], ORCA_RESULT_FILES, kwargs["arg_hash"])


@node
async def orca(qm_input: QMInput, **kwargs) -> SimstackResult:
    """Async function that generates input files, runs a configuration-driven
    ORCA computation workflow, and parses the results.

    This node mirrors :func:`orca` in terms of input generation, safety
    checks, and output parsing, but constructs the ORCA execution command
    from the project-level ``config.toml`` via
    :func:`_build_orca_command_from_config`. If no suitable configuration
    is found (for example, if ``config.toml`` is missing or does not
    define a ``[<resource>.program.orca]`` section), it transparently
    falls back to the same hard-coded commands used by :func:`orca` for
    local and cluster resources.

    Input preparation is based on the parameters provided in the
    ``qm_input`` object and supports a broad range of ORCA settings,
    including basis set, functional, spin multiplicity, solvation model,
    and advanced correlation methods. Depending on the requested method
    and options, this node constructs the corresponding ORCA input
    blocks (e.g. CASSCF, TDDFT, SCF, geometry optimisation, gradients),
    handles constraints, and writes the full ``orca.inp`` file.

    Similar to :func:`orca`, it performs safety checks for restart and
    supporting files (e.g. ``orca.xyz``, ``orca.gbw`` or user-provided
    restart files) and incorporates them into the calculation setup. If a
    calculation that used restart data fails (non-zero return code or
    abnormal termination), it automatically retries once from the molecule
    geometry without restart data. It also logs effective Slurm parameters
    and other configuration details via :class:`NodeRunner` for debugging
    and auditing.

    After the ORCA subprocess finishes, the node parses the resulting
    ``orca.out`` file to extract ground-state and excited-state
    information. This includes, where available, final structures,
    excited states and state transitions, absorption spectra, vibrational
    frequencies, normal modes, IR spectra, and Mayer bond analysis. The
    parsed data are assembled into a :class:`QMResult` instance and
    attached to the node's :class:`SimstackResult`.

    If hyperpolarizability properties are requested via
    ``qm_input.elprop.Hyperpol``, the node automatically ensures that the
    dipole moment calculation is enabled (``elprop.Dipole = True``) and
    upgrades the SCF accuracy to at least ``Tight`` for more reliable
    hyperpolarizability results, matching the behaviour of
    :func:`orca`.

    Parameters:
        qm_input (QMInput): Quantum mechanical input parameters object
            that specifies molecular, electronic, and computational
            details for ORCA calculations.

    Returns:
        SimstackResult: Parsed result from the ORCA calculations
        containing both ground-state and excited-state information,
        extracted through standard and customised parsing for state
        transitions, absorption spectra, vibrational data, and Mayer
        analysis.

    SimstackResult:
        orca_result (QMResult): Parsed result from the ORCA
            calculations.
        orca_elprop_result (QMResultElProp, optional): Electronic
            properties result derived from the same ORCA output.
        files (List[FileStack]): List of files generated during
            execution of the node (such as ``orca.inp``, ``orca.out`` and
            related auxiliary files).

    Raises:
        Exception: If there is a failure in generating input files,
        executing the ORCA subprocess, or parsing the ORCA output, the
        function logs and propagates a detailed error via
        :class:`NodeRunner`.
    """


    node_runner = kwargs["node_runner"]

    used_restart = await _write_orca_input(qm_input, use_restart=True, **kwargs)
    node_runner.info_files.append(FileStack.from_local_file("orca.inp", in_memory=True, is_hashable=True, secure_source=True))
    node_runner.info("input files done")

    input_files = ["orca.inp", "orca.gbw"]
    returncode = orca_run_command(input_files, ORCA_RESULT_FILES, kwargs["arg_hash"])

    if returncode != 0 and used_restart:
        returncode = await _retry_orca_without_restart(
            qm_input,
            f"orca execution failed with return code {returncode} after using restart data",
            **kwargs,
        )
        used_restart = False

    if returncode != 0:
        return node_runner.fail(f"orca execution failed with return code {returncode}")

    orca_run = None
    orca_out_contents = None
    try:
        if os.path.exists("orca.out"):
            # Cache the full ORCA output directly after the run so that later
            # post-processing does not rely on the file still being present on
            # disk. Do this before FileStack operations.
            with open("orca.out", "r", encoding="utf-8", errors="ignore") as f:
                orca_out_contents = f.read()

            node_runner.info_files.append(
                FileStack.from_local_file(
                    "orca.out", in_memory=True, is_hashable=True, secure_source=True
                )
            )

            orca_run = OrcaOutput(node_runner, "orca.out")
            if not orca_run.normal_termination:
                tolerated = _is_tolerated_abnormal_termination(qm_input, orca_out_contents)
                if used_restart and not tolerated:
                    returncode = await _retry_orca_without_restart(
                        qm_input,
                        "orca did not terminate normally after using restart data",
                        **kwargs,
                    )
                    if returncode != 0:
                        return node_runner.fail(
                            f"orca execution failed with return code {returncode}"
                        )
                    if not os.path.exists("orca.out"):
                        return node_runner.fail("orca.out file not found")
                    with open("orca.out", "r", encoding="utf-8", errors="ignore") as f:
                        orca_out_contents = f.read()
                    node_runner.info_files.append(
                        FileStack.from_local_file(
                            "orca.out", in_memory=True, is_hashable=True, secure_source=True
                        )
                    )
                    orca_run = OrcaOutput(node_runner, "orca.out")
                    if not orca_run.normal_termination:
                        if _is_tolerated_abnormal_termination(qm_input, orca_out_contents):
                            node_runner.warning(
                                "orca did not terminate normally but it was tolerated. Continuing execution."
                            )
                        else:
                            return node_runner.fail("orca did not terminate normally")
                elif tolerated:
                    node_runner.warning(
                        "orca did not terminate normally but it was tolerated. Continuing execution."
                    )
                else:
                    return node_runner.fail("orca did not terminate normally")
            node_runner.info("calculation finished")
        else:
            node_runner.fail("orca.out file not found")
    except Exception as e:
        return node_runner.fail(f"orca.out parsing error {str(e)}")

    try:
        if orca_run is not None:
            orca_result = await QMResult.from_orca_output(orca_run, node_runner.task_id)

            try:
                dipole_val = getattr(orca_run, "dipole", None)
                dipole_moment_val = getattr(orca_run, "dipole_moment", None)
                node_runner.info(
                    f"ORCA result electronic properties: dipole={dipole_val}, dipole_moment={dipole_moment_val}"
                )
                hyper_info = getattr(orca_run, "_hyperpolarizability", None)
                node_runner.info(
                    f"ORCA hyperpolarizability mapping present: {hyper_info is not None}"
                )
            except Exception as e_el:  # pragma: no cover - debug logging only
                node_runner.warning(
                    f"Failed to log ORCA/QMResult electronic properties: {e_el}"
                )

            if orca_result.final_structure:
                orca_result.final_structure.smiles = qm_input.molecule.smiles
                orca_result.final_structure.formula = qm_input.molecule.formula
            if orca_result.structures is not None:
                for molecule in orca_result.structures:
                    molecule.smiles = qm_input.molecule.smiles
                    molecule.formula = qm_input.molecule.formula

            node_runner.info("done standard ORCA parsing")

            # Use cached orca.out contents from immediately after the run.
            contents = orca_out_contents
            try:
                # Parse orbital energies
                orbital_energies_df = parse_orbital_energies(contents, is_filename=False)
                logger.info(f"Parsed orbital energies DataFrame: {orbital_energies_df.head() if orbital_energies_df is not None else 'None'}")
            except Exception as e:
                node_runner.error(f"Error parsing ORCA orbital energies: {str(e)}")
                orbital_energies_df = None

            # Update the existing QMResult instance with orbital energies
            if orbital_energies_df is not None:
                try:
                    orca_result.set_values_from_orbital_energies_dataframe(orbital_energies_df)
                    node_runner.info("Orbital energies parsed and set on QMResult (orca_jinja node)")
                except Exception as e_orb:  # pragma: no cover - defensive
                    node_runner.warning(f"Failed to set orbital energies on QMResult: {e_orb}")

            try:
                states_table, transition_table = parse_orca_excited_states(contents)
                if states_table and len(states_table.row) > 0:
                    orca_result.excited_states = states_table
                if transition_table and len(transition_table.row) > 0:
                    orca_result.excited_state_transitions = transition_table
                node_runner.info("done ORCA excited states parsing")
            except Exception as e:
                node_runner.error(f"Error parsing ORCA excited states: {str(e)}")

            try:
                absorption_spectrum = parse_orca_absorption_spectrum(contents)
                if absorption_spectrum and len(absorption_spectrum.row) > 0:
                    orca_result.absorption_spectrum = absorption_spectrum
                node_runner.info("done ORCA absorption spectrum parsing")
            except Exception as e:
                node_runner.error(f"Error parsing ORCA absorption spectrum: {str(e)}")

            try:
                mayer_analysis, mayer_bond_orders = parse_mayer_analysis(contents)
                if mayer_analysis and len(mayer_analysis.row) > 0:
                    orca_result.mayer_analysis = mayer_analysis
                if mayer_bond_orders and len(mayer_bond_orders.row) > 0:
                    orca_result.mayer_bond_orders = mayer_bond_orders
                node_runner.info("done ORCA mayer analysis parsing")
            except Exception as e:
                node_runner.error(f"Error parsing ORCA mayer analysis: {str(e)}")

            try:
                vibrational_frequencies = parse_vibrational_frequencies(contents)
                if vibrational_frequencies and len(vibrational_frequencies.row) > 0:
                    orca_result.vibrational_frequencies = vibrational_frequencies
                normal_modes = parse_normal_modes(contents)
                if normal_modes and len(normal_modes.row) > 0:
                    orca_result.normal_modes = normal_modes
                ir_spectrum = parse_ir_spectrum(contents)
                if ir_spectrum and len(ir_spectrum.row) > 0:
                    orca_result.ir_spectrum = ir_spectrum
                node_runner.info("done ORCA vibrational frequencies parsing")
            except Exception as e:
                node_runner.error(f"Error parsing ORCA vibrational frequencies: {str(e)}")
        else:
            return node_runner.fail("orca_run is none")

        # Construct and attach dedicated electronic-properties result
        # directly from the OrcaOutput instance.
        try:
            elprop_result = QMResultElProp.from_orca_output(
                orca_run,
                parent_qm_result=orca_result,
                task_id=node_runner.task_id,
            )
            node_runner.orca_elprop_result = elprop_result
            node_runner.info("QMResult_elprop created from OrcaOutput (orca_jinja node)")

            # TODO: verbose debug logging of hyperpolarizability tensors.
            # This is for debugging only and can be disabled later by
            # switching the condition to ``if False``.
            if False: #debug deactivated - switch to True if stuff seems to be missing again
                try:
                    node_runner.info(
                        "QMResult_elprop.static_hyperpolarizability_tensor = %s",
                        elprop_result.static_hyperpolarizability_tensor,
                    )
                    node_runner.info(
                        "QMResult_elprop.aligned_static_hyperpolarizability_tensor = %s",
                        elprop_result.aligned_static_hyperpolarizability_tensor,
                    )

                except Exception as e_elprop_debug:  # pragma: no cover - debug logging only
                    node_runner.warning(
                        "Failed to log QMResult_elprop hyperpolarizability tensors: %s",
                        e_elprop_debug,
                    )
        except Exception as e_elprop:  # pragma: no cover - defensive
            node_runner.warning(f"Failed to construct QMResult_elprop from OrcaOutput: {e_elprop}")

        node_runner.orca_result = orca_result
        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error reading ORCA result: {str(e)}")
