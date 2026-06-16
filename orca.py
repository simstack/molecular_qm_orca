import os
from typing import List
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
from molecular_qm_orca.deprecated.orca_output import OrcaOutput


def orca_run_command(input_files: List[str], result_files: List[str],arg_hash: str):
    return context.config.resource_config.run("orca", input_files, result_files)

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
    restart files) and incorporates them into the calculation setup. It
    also logs effective Slurm parameters and other configuration details
    via :class:`NodeRunner` for debugging and auditing.

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
        orca_elprop_result (QMResult_elprop, optional): Electronic
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

    orca_input_gen = orca_input_factory(qm_input, **kwargs)

    first_line = orca_input_gen.first_line
    blocks = orca_input_gen.blocks

    with open("orca.inp", "w") as f:
        f.write(first_line)
        for block in blocks:
            f.write(block)

    node_runner.info_files.append(FileStack.from_local_file("orca.inp", in_memory=True, is_hashable=True, secure_source=True))
    node_runner.info("input files done")

    input_files = ["orca.inp", "orca.gbw"]
    result_files = ["orca.out", "orca.gbw", "orca.xyz","orca.trj", "orca.densities", "orca.engrad", "orca.opt",
                    "orca.property", "orca_run.log"]
    result = orca_run_command(input_files, result_files, kwargs["arg_hash"])

    if result.returncode != 0:
        return node_runner.fail(f"orca execution failed with return code {result.returncode}")


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

            path_without_extension = "orca"
            orca_run = OrcaOutput(path_without_extension)
            if not orca_run.normal_termination:
                if qm_input.tolerate_failure:
                    with open("orca.out", "r") as f:
                        orca_out_contents = f.read()
                    msg = "unfortunately, the SCF has not converged. There may be a way out but we have to stop here"
                    if msg not in orca_out_contents:
                        return node_runner.fail("orca did not terminate normally")
                    else:
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
            async for molecule in orca_result.structures:
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
