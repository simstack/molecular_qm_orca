import os
import subprocess

from molecular_qm_orca.pyorca import OrcaInput, OrcaRun
from .orbital_energies_parser import parse_orbital_energies
from .orca_absorption_spectrum_parser import parse_orca_absorption_spectrum
from .orca_excited_states_parser import parse_orca_excited_states
from .orca_frequency_parser import (
    parse_vibrational_frequencies,
    parse_normal_modes,
    parse_ir_spectrum,
)
from .orca_mayer_parser import parse_mayer_analysis
from simstack.core.simstack_result import SimstackResult
from molecular_qm_models import BasisSet, AuxBasis, BasisSetEnum, AuxBasisEnum
from molecular_qm_models import Functional
from molecular_qm_models import DispersionCorrection
from molecular_qm_models import Molecule, Atom
from molecular_qm_models import QMResult, QMResult_elprop

from molecular_qm_models import (
    QMInput,
    SCFAccuracy,
    OptimizationAccuracy,
)
from simstack.core.node import node
from simstack.core.node_runner import NodeRunner
from simstack.models import Parameters
from simstack.core.context import context
import logging

from simstack.models.files import FileStack
from simstack.models.parameters import SlurmParameters

# Piecewise ORCA workflows (input-only, run-only, result-collection and
# combined helpers) live in a dedicated module to keep this file focused on
# the monolithic nodes. We re-export the nodes here so existing imports from
# ``applications.electronic_structure.orca.orca`` continue to work.

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - fallback for older interpreters
    import tomli as tomllib
#https://docs.python.org/3/library/tomllib.html
#tomllib conversion table
#TOML document > dict
#string > str
#integer > int
#float > float
#boolean > bool
# offset date-time > datetime.datetime (tzinfo attribute set to an isntance of datetime.timezone   )
# local date-time > datetime.datetime (tzinfo attribute set to None)
# local date > datetime.date
# local time > datetime.time
# array > list
# table > dict
# inline table > dict
# array of tables > list of dicts

logger = logging.getLogger("OrcaNode")


def multiplicity_to_string(multiplicity: int) -> str:
    table = ["singlet", "doublet", "triplet", "quadruplet", "quintuplet"]
    if multiplicity < 1 or multiplicity > 5:
        raise ValueError(f"Invalid multiplicity {multiplicity}. Valid values are 1 to 5.")
    return table[multiplicity - 1]


def _orca_frequency_keyword(qm_input: QMInput) -> str | None:
    """Return the ORCA frequency keyword for AUTO behavior."""
    if not qm_input.frequencies:
        return None

    # Methods for which analytical frequencies are typically available in ORCA.
    # If the method is not listed here, we fall back to numerical frequencies.
    analytical_methods = {
        "HF",
        "DFT",
        "MP2",
        "CCSD",
        "CCSD(T)",
        "CIS",
        "RPA",
    }

    if qm_input.method.value in analytical_methods:
        return "FREQ"

    return "NUMFREQ"

#TODO orca fails with no error when run on the default queue

slurm_parameters = SlurmParameters()
#no overrides!!! those are only for testing and confuse the user
#slurm_parameters.time = "2:00:00"
#slurm_parameters.mem = "2G"
#slurm_parameters.cpus_per_task = 8
parameters = Parameters(resource="int-nano", queue="slurm-queue", slurm_parameters=slurm_parameters)
local_parameters = Parameters(resource="local")


def _build_orca_command_from_config(resource: str, program_name: str = "orca") -> str | None:
    """Build the ORCA run command from ``config.toml`` for the given resource.

    This helper reads the project-level ``config.toml`` (as used by the
    Simstack runner) and looks for a section of the form

    .. code-block:: toml

        [<resource>.program.orca]
        scripts = ["..."]      # optional
        environment_modules = ["..."]  # optional
        run_command = "..."    # required

    The returned string is a single shell command where all configured
    ``scripts`` / ``environment_modules`` are chained together using ``&&``
    and finally followed by ``run_command``.

    If anything is missing (no config file, no matching resource/program,
    no run_command), ``None`` is returned so callers can fall back to the
    legacy hard-coded behaviour.
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

    # Optional environment modules (e.g. ["orca/6.1.1"]) – translated into
    # simple "module load <name>" lines. This mirrors the examples in
    # ``config.toml`` while keeping the implementation minimal.
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

####depreciated- one should use the jinja nodes instead - no longer tested after the migration to the new QMinput
@node(parameters=parameters)
async def orca(qm_input: QMInput, **kwargs) -> SimstackResult:
    """
    Async function that generates input files, runs a computation workflow using ORCA quantum chemistry software,
    and parses the results.

    This function prepares input files based on the parameters provided in the `qm_input` object for quantum
    chemistry calculations using ORCA. It supports various configurations, such as basis set, functional,
    spin multiplicity, solvent model, and more. Intermediate input details, constraints, and specialized
    blocks, like CASSCF or TDDFT configurations, are generated and written to the input file. It ensures
    safety checks for existing supporting files (e.g., `orca.xyz`, `orca.gbw`) and incorporates them into
    calculation setup. The function handles the actual subprocess call for executing ORCA and parses the
    resulting output file to extract relevant quantum chemistry results.

    The results include final structures, state transitions, excited-state data, and optionally absorption
    spectra, all wrapped into a `SimstackResult` object. The method logs details of its workflow for debugging
    or audits and captures errors at various stages.

    Parameters:
        qm_input (QMInput): Quantum mechanical input parameters object that specifies molecular,
        electronic, and computational details for ORCA calculations.
        **kwargs: Additional arbitrary keyword arguments used for configuration or overriding specific
        functionalities. Commonly includes `parameters` specifying ORCA runtime configurations.

    Returns:
        SimstackResult: Parsed result from the ORCA calculations containing both ground-state and excited-state
        information, extracted through standard and customized parsing for state transitions and absorption
        spectra.

    SimstackResult:
        orca_result (QMResult): Parsed result from the ORCA calculations
        files (List[FileStack]): List of files generated during the execution of the node

    Raises:
        Exception: If there is a failure in generating input files, execution of subprocess, or parsing
        ORCA output, the function handles and raises detailed exceptions for errors.
    """
    #the following imports need to be imported to the orca CLASS not somewhere else- limits the scope AND allows later for making it version specific!
    from .orca_main_lib import (
        add_grid_to_simple_input_line,
        add_tddft_block_if_needed,
        add_electronic_properties_block,
        make_casscf_block,
        use_gbw_restart,
        set_orca_memory_and_pal_options_according_to_slurm_parameters,
        set_method_and_basis_set_for_non_casscf_methods,
        set_orca_optimization_options,
    )
    node_runner = NodeRunner("orca", logger=logger, **kwargs)

    # Log effective Slurm parameters actually seen by the ORCA node. This
    # helps debug situations where many nested ORCA jobs time out because
    # they don't inherit the expected time/memory/core settings.
    try:
        local_params = kwargs.get("parameters", None)
        parent_params = kwargs.get("parent_parameters", None)
        if local_params is None:
            try:
                local_params = orca._node_parameters
            except AttributeError:
                local_params = None

        def _log_slurm_params(label: str, params_obj) -> None:
            if params_obj is None:
                node_runner.info(f"[SLURM] {label}: parameters=None")
                return
            slurm = getattr(params_obj, "slurm_parameters", None)
            node_runner.info(
                "[SLURM] %s: resource=%s, queue=%s, nodes=%s, tasks_per_node=%s, cpus_per_task=%s, mem=%s, time=%s"
                % (
                    label,
                    getattr(params_obj, "resource", None),
                    getattr(params_obj, "queue", None),
                    getattr(slurm, "nodes", None) if slurm else None,
                    getattr(slurm, "tasks_per_node", None) if slurm else None,
                    getattr(slurm, "cpus_per_task", None) if slurm else None,
                    getattr(slurm, "mem", None) if slurm else None,
                    getattr(slurm, "time", None) if slurm else None,
                )
            )

        _log_slurm_params("EFFECTIVE", local_params)
        _log_slurm_params("PARENT", parent_params)
    except Exception:
        # Do not let logging issues break the ORCA job
        pass
    try:
        node_runner.info(f"multiplicity: {qm_input.multiplicity} optimize: {qm_input.optimization}")
        node_runner.info(f"charge: {qm_input.charge}")
        node_runner.info(f"active_electrons: {qm_input.active_electrons} active_orbitals: {qm_input.active_orbitals}")
        node_runner.info(f"solvent: {qm_input.solvent}")

        # Ensure that hyperpolarizability requests have the necessary
        # supporting settings. Specifically:
        #   * If elprop.Hyperpol is True, force elprop.Dipole = True so that
        #     a dipole moment is available for tensor alignment.
        #   * If elprop.Hyperpol is True, enforce at least Tight SCF
        #     convergence (upgrade Sloppy/Loose/Medium/Strong -> Tight).
        try:
            elprop_cfg = getattr(qm_input, "elprop", None)
            if elprop_cfg is not None and getattr(elprop_cfg, "Hyperpol", False):
                # Dipole moment is required for alignment of the
                # hyperpolarizability tensor; silently upgrade and log.
                if not getattr(elprop_cfg, "Dipole", False):
                    node_runner.info(
                        "ElProp Hyperpol requested but Dipole=False; overriding elprop.Dipole -> True "
                        "to ensure dipole moment is available for hyperpolarizability alignment."
                    )
                    elprop_cfg.Dipole = True

                # Enforce at least Tight SCF accuracy for Hyperpol jobs.
                if qm_input.scf_accuracy in {
                    SCFAccuracy.Sloppy,
                    SCFAccuracy.Loose,
                    SCFAccuracy.Medium,
                    SCFAccuracy.Strong,
                }:
                    old_acc = qm_input.scf_accuracy
                    qm_input.scf_accuracy = SCFAccuracy.Tight
                    node_runner.info(
                        f"ElProp Hyperpol requested with scf_accuracy={old_acc.value}; "
                        "overriding to 'Tight' for more reliable hyperpolarizability."
                    )
        except Exception as e_over:
            # Do not let safety overrides break the job; just log.
            node_runner.warning(f"Failed to apply Hyperpol-dependent overrides: {e_over}")
        blocks = set_orca_memory_and_pal_options_according_to_slurm_parameters(
            node_runner,
            blocks=None,
            kwargs=kwargs,
            node_slurm_params=orca._node_parameters.slurm_parameters,
        )

        if qm_input.active_electrons > 0:  # casscf

            first_line, cas_scf_spec_bloc, cas_scf_output_block = make_casscf_block(qm_input)
            blocks.append(cas_scf_spec_bloc)
            blocks.append(cas_scf_output_block)
        #TODO elif option for gold standard method like CCSD(T) can be added here in the future with its own specific blocks and first line specifications
        # this can then also use refactored grid etc mothods from the lib
        # SIMILARLY HF or similar if desired (not a priority )
        #else- defaults to dft!
        else:

            aux_basis = qm_input.basis_set.aux_basis.aux_basis.value if qm_input.basis_set.aux_basis.aux_basis != AuxBasisEnum.NONE else ""
            first_line = set_method_and_basis_set_for_non_casscf_methods(
                node_runner,
                qm_input,
                aux_basis,
                first_line="",
            )
            if qm_input.multiplicity > 1 or qm_input.open_shell_calculation:
                first_line += " UHF"

            # Centralised handling of geometry optimisation accuracy and %geom block
            first_line, blocks = set_orca_optimization_options(node_runner, qm_input, first_line, blocks)
            if not getattr(qm_input, "optimization", False) and qm_input.gradients:
                first_line += " ENGRAD"

            node_runner.info(f"frequencies: {qm_input.frequencies}")
            freq_keyword = _orca_frequency_keyword(qm_input)
            if freq_keyword:
                first_line += f" {freq_keyword}"

            if qm_input.solvent.lower() != "none":
                first_line += f" {qm_input.solvent_model.value}({qm_input.solvent.upper()})"

            #adds grid if specified in qm_input. This is done in the main lib because it is relevant for multiple methods and will be modified version specific in the future
            # furthermore it declutters the main
            first_line = add_grid_to_simple_input_line(first_line, qm_input)

            blocks = add_electronic_properties_block(qm_input, blocks)


                            # Map scf_accuracy enum to ORCA Convergence keywords
            conv_keyword = qm_input.scf_accuracy.value

            opt_conv_keyword = qm_input.optimization_accuracy.value

            scf_block = f"%scf\n Convergence {conv_keyword}\n MaxIter {qm_input.max_scf_iterations}\nend\n\n"
            blocks.append(scf_block)

            if qm_input.optimization:
                geom_block = f"%geom\n Convergence {opt_conv_keyword}\n MaxIter {qm_input.max_optimization_iterations}\nend\n\n"
                blocks.append(geom_block)

            # Only add TDDFT block if states > 0


            blocks = add_tddft_block_if_needed(qm_input, blocks)
        

        # Check if there are constraints to apply
        constraints = qm_input.molecule.properties.get("constraints", [])
        node_runner.info(f"constraints: {constraints}")
        if constraints:
            from applications.constraints import MolecularConstraintType
            geom_block = "%geom\n  Constraints\n"
            for constraint in constraints:
                # constraints might be a list of dicts or MolecularConstraint objects
                if isinstance(constraint, dict):
                    c_type = constraint.get("type")
                    c_atoms = constraint.get("atom_indices")
                else:
                    c_type = constraint.type
                    c_atoms = constraint.atom_indices

                if c_type == MolecularConstraintType.DISTANCE:
                    geom_block += f"    {{B {c_atoms[0]} {c_atoms[1]} C}}\n"
                elif c_type == MolecularConstraintType.ANGLE:
                    geom_block += f"    {{A {c_atoms[0]} {c_atoms[1]} {c_atoms[2]} C}}\n"
                elif c_type == MolecularConstraintType.DIHEDRAL or c_type == MolecularConstraintType.IMPROPER:
                    geom_block += f"    {{D {c_atoms[0]} {c_atoms[1]} {c_atoms[2]} {c_atoms[3]} C}}\n"
                elif c_type == MolecularConstraintType.FROZEN:
                    for atom_idx in c_atoms:
                        geom_block += f"    {{C {atom_idx} C}}\n"

            geom_block += "  end\nend\n\n"
            blocks.append(geom_block)




        # Parallelization and memory based on Slurm parameters
        parent_params = kwargs.get("parent_parameters")
        if parent_params and parent_params.slurm_parameters and parent_params.queue == slurm_parameters:
            slurm = parent_params.slurm_parameters
            tasks = slurm.tasks
            if tasks is None:
                tasks = 1
            nodes = slurm.nodes
            if nodes is None:
                nodes = 1
            tasks_per_node = slurm.tasks_per_node
            if tasks_per_node is None:
                tasks_per_node = 1
            node_runner.info(f"Slurm tasks: {tasks}, nodes: {nodes}, tasks_per_node: {tasks_per_node}")
            # If tasks is not explicitly set, calculate it as nodes * tasks_per_node
            if tasks == 1 and (nodes > 1 or tasks_per_node > 1):
                tasks = nodes * tasks_per_node

            if tasks > 1:
                pal_block = f"%pal\n nprocs {tasks}\nend\n\n"
                blocks.append(pal_block)
                node_runner.info(f"Adding parallelization block: {tasks} processes")

            # Memory handling: Slurm's 'mem' is total memory per node,
            # ORCA's %maxcore is memory per process in MB.
            if slurm.mem:
                try:
                    mem_str = slurm.mem.upper().strip()
                    if mem_str.endswith("G"):
                        total_mem_mb = int(float(mem_str[:-1]) * 1024)
                    elif mem_str.endswith("M"):
                        total_mem_mb = int(float(mem_str[:-1]))
                    elif mem_str.endswith("MB"):
                        total_mem_mb = int(float(mem_str[:-2]))
                    elif mem_str.endswith("GB"):
                        total_mem_mb = int(float(mem_str[:-2]) * 1024)
                    else:
                        # try to parse as float directly
                        total_mem_mb = int(float(mem_str)) # Assume MB if no unit

                    # Memory per process (MB)
                    # We use tasks_per_node if it's a multi-node job, or nprocs if it's single-node
                    tasks_on_this_node = slurm.tasks_per_node if slurm.nodes > 1 else tasks
                    if tasks_on_this_node > 0:
                        # ORCA manual recommends about 75% of available RAM to avoid swapping
                        maxcore = int((total_mem_mb * 0.75) / tasks_on_this_node)
                        if maxcore > 0:
                            blocks.append(f"%maxcore {maxcore}\n\n")
                            node_runner.info(f"Setting %maxcore to {maxcore} MB per process")
                except (ValueError, TypeError):
                    node_runner.warning(f"Could not parse Slurm memory: {slurm.mem}")

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
                first_line, blocks = use_gbw_restart(node_runner, qm_input, first_line, blocks)

        if qm_input.first_line is not None and qm_input.first_line != "":
            first_line += f" {qm_input.first_line}"
        first_line += "\n\n"
        params = [first_line]

        orca_input_file = OrcaInput(qm_input.molecule, charge=qm_input.charge,
                                    spin_multiplicity=qm_input.multiplicity,
                                    input_parameters=params, blocks=blocks, with_geometry=with_geometry)

        with open("orca.inp", "w") as f:
            f.write(str(orca_input_file))
            #for debug reasons also write this ot the console log
            node_runner.info(f"orca.inp contents:\n{str(orca_input_file)}")
        node_runner.info_files.append(FileStack.from_local_file("orca.inp",in_memory=True, is_hashable=True, secure_source=True))
        node_runner.info("input files done")
    except Exception as e:
        return node_runner.fail(f"Error generating input files {str(e)}")

    resource = kwargs["parent_parameters"].resource

    shell = ""
    if resource == "local" or resource == "self" or resource == "local-home":
        # Use subprocess to determine the user's shell. If it's bash use `which`, otherwise use `where`.
        try:
            shell_check = subprocess.run(["/bin/sh", "-c", "echo $SHELL"], capture_output=True, text=True, check=False)
            shell = (shell_check.stdout or "").strip()
        except Exception:
            # Fall back to the environment variable if the subprocess check fails
            shell = os.environ.get("SHELL", "")

        # If the detected shell contains 'bash' use 'which', otherwise use 'where'
        if shell and "bash" in shell:
            locator_cmd = "which"
        else:
            locator_cmd = "where"

        command = f"{locator_cmd} orca && orca orca.inp > orca.out"
        # WW command = "C:\\ORCA_6.1.1\\orca.exe orca.inp > orca.out"
    elif resource == "int-nano":
        #try:
        #except block here later to do this new version as a default and fall back to older versions otherwise
            #first try the most recent module orca/6.1.1 if it
        try:
            shell_check = subprocess.run(["/bin/sh", "-c", "echo $SHELL"], capture_output=True, text=True, check=False)
            shell = (shell_check.stdout or "").strip()
        except Exception:
            # Fall back to environment variable if subprocess check fails
            shell = os.environ.get("SHELL", "")

        # If the detected shell contains 'bash' use 'which', otherwise use 'where'
        if shell and "bash" in shell:
            locator_cmd = "which"
        else:
            locator_cmd = "where"

        command = f"{locator_cmd} orca && orca orca.inp > orca.out"
    elif resource == "int-nano" or resource =="int-nano-jinja":

        #first try the most recent module orca/6.1.1 if it - config later via jinja template
        #TODO jinja
        #command = "source ~/.bashrc && module load orca && module load openmpi && $ORCA_HOME/orca orca.inp > orca.out"
        #PLEASE NO INDIVIDUAL OPENMPI LOADING_ THIS IS BECAUSE THE ORCA MODULE LOADS THE CORRECT VERSION OF OPENMPI AS A DEPENDENCY. LOADING ANOTHER VERSION CAUSES ISSUES
        #using orca home env variable is unsafe! better would be runner=$(which orca); $runner orca.inp > orca.out for bash. We keep this so far
        #command = "source ~/.bashrc && module load orca/6.1.1 && $ORCA_HOME/orca orca.inp > orca.out"
        command = "source ~/.bashrc && module load orca/6.1.1 && $(which orca) orca.inp > orca.out"
        #why the source .bashrc here? that could add other stuff to the path and LD_LIBRARY_PATH that could potentially conflict for individual users


    else:
        return node_runner.fail(f"Unsupported resource {resource}")

    node_runner.info_files.append(
        FileStack.from_local_file("orca.inp", in_memory=True, is_hashable=True, secure_source=True))

    # subprocess will fail if not converged
    if not node_runner.subprocess("orca_run",command):
        if not qm_input.tolerate_failure:
            if os.path.exists("orca.out"):
                node_runner.info_files.append(
                    FileStack.from_local_file("orca.out",       in_memory=True, is_hashable=True, secure_source=True))
            if os.path.exists("orca_run.log"):
                node_runner.info_files.append(
                    FileStack.from_local_file("orca_run.log", in_memory=True, is_hashable=True, secure_source=True))
            return node_runner.fail(f"Error running ORCA calculation {node_runner.last_stderr}")

    orca_run = None
    orca_out_contents = None
    try:
        if os.path.exists("orca.out"):
            # Cache full ORCA output immediately after the run so we don't
            # depend on the file still being present hours later during
            # post-processing. Do this *before* creating the FileStack to
            # avoid any side effects of FileStack I/O.
            with open("orca.out", "r", encoding="utf-8", errors="ignore") as f:
                orca_out_contents = f.read()

            node_runner.info_files.append(
                FileStack.from_local_file(
                    "orca.out", in_memory=True, is_hashable=True, secure_source=True
                )
            )

            # get the filename without the extension
            path_without_extension = "orca"
            orca_run = OrcaRun(path_without_extension, node_runner=node_runner)
            if not orca_run.normal_termination:
                if qm_input.tolerate_failure:
                    with open("orca.out", "r", encoding='utf-8', errors='replace') as f:
                        orca_out_contents = f.read()
                    if  "unfortunately, the SCF has not converged. There may be a way out but we have to stop here" not in orca_out_contents:
                        return node_runner.fail("orca did not terminate normally")
                    else:
                        node_runner.warning("orca did not terminate normally but it was tolerated. Continuing execution.")
                else:
                    return node_runner.fail("orca did not terminate normally")
            node_runner.info("calculation finished")
        else:
            return node_runner.fail("orca.out file not found")
    except Exception as e:
        return node_runner.fail(f"orca.out parsing error {str(e)}")

    try:
        if orca_run is not None:
            orca_result = await QMResult.from_orca_output(orca_run, node_runner.task_id)

            # Temporary logging of electronic properties / hyperpolarizability so
            # we can see in the node log what ORCA produced and what made it
            # into the QMResult. This is intentionally verbose and can be
            # removed or downgraded once things are confirmed to work.
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
                # Same here: format manually rather than passing extra args.
                node_runner.warning(
                    f"Failed to log ORCA/QMResult electronic properties: {e_el}"
                )

            if orca_result.final_structure:
                orca_result.final_structure.smiles = qm_input.molecule.smiles
                orca_result.final_structure.formula = qm_input.molecule.formula
            for molecule in orca_result.structures.molecules:
                molecule.smiles = qm_input.molecule.smiles
                molecule.formula = qm_input.molecule.formula

            node_runner.info("done standard ORCA parsing")

            with open("orca.out", encoding='utf-8', errors='replace') as f:
                contents = f.read()
            # Use the cached orca.out contents captured immediately after the
            # ORCA run. This avoids depending on the file still being present
            # on disk at this late stage.
            
            contents = orca_out_contents
            try:
                # Parse orbital energies
                orbital_energies_df = parse_orbital_energies(contents, is_filename=False)
            except Exception as e:
                node_runner.error(f"Error parsing ORCA orbital energies: {str(e)}")
                orbital_energies_df = None

            # Update the existing QMResult instance with orbital energies
            if orbital_energies_df is not None:
                try:
                    orca_result.set_values_from_orbital_energies_dataframe(orbital_energies_df)
                    node_runner.info("Orbital energies parsed and set on QMResult (orca node)")
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
                # Parse all three sections using the same content
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

        # Build a separate electronic-properties result object from the
        # raw OrcaRun output. This keeps elprop parsing encapsulated in
        # QMResult_elprop and decouples it from the core QMResult model.
        try:
            elprop_result = QMResult_elprop.from_orca_output(
                orca_run,
                parent_qm_result=orca_result,
                task_id=node_runner.task_id,
            )
            node_runner.orca_elprop_result = elprop_result
            node_runner.info("QMResult_elprop created from OrcaRun (orca node)")
        except Exception as e_elprop:  # pragma: no cover - defensive
            node_runner.warning(f"Failed to construct QMResult_elprop from OrcaRun: {e_elprop}")

        node_runner.orca_result = orca_result
        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error reading ORCA result: {str(e)}")


@node(parameters=parameters)
async def orca_jinja(qm_input: QMInput, **kwargs) -> SimstackResult:
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
        **kwargs: Additional arbitrary keyword arguments used for
            configuration or overriding specific functionalities. These
            typically include ``parameters`` / ``parent_parameters``
            describing runtime resources (e.g. Slurm configuration).

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

    from molecular_qm_orca.orca_main_lib import (
        add_grid_to_simple_input_line,
        add_tddft_block_if_needed,
        add_electronic_properties_block,
        make_casscf_block,
        use_gbw_restart,
        set_orca_memory_and_pal_options_according_to_slurm_parameters,
        set_method_and_basis_set_for_non_casscf_methods,
    )

    node_runner = NodeRunner("orca_jinja", logger=logger, **kwargs)

    # Log effective Slurm parameters actually seen by the ORCA node.
    try:
        local_params = kwargs.get("parameters", None)
        parent_params = kwargs.get("parent_parameters", None)
        if local_params is None:
            try:
                local_params = orca_jinja._node_parameters
            except AttributeError:
                local_params = None

        def _log_slurm_params(label: str, params_obj) -> None:
            if params_obj is None:
                node_runner.info(f"[SLURM] {label}: parameters=None")
                return
            slurm = getattr(params_obj, "slurm_parameters", None)
            node_runner.info(
                "[SLURM] %s: resource=%s, queue=%s, nodes=%s, tasks_per_node=%s, cpus_per_task=%s, mem=%s, time=%s"
                % (
                    label,
                    getattr(params_obj, "resource", None),
                    getattr(params_obj, "queue", None),
                    getattr(slurm, "nodes", None) if slurm else None,
                    getattr(slurm, "tasks_per_node", None) if slurm else None,
                    getattr(slurm, "cpus_per_task", None) if slurm else None,
                    getattr(slurm, "mem", None) if slurm else None,
                    getattr(slurm, "time", None) if slurm else None,
                )
            )

        _log_slurm_params("EFFECTIVE", local_params)
        _log_slurm_params("PARENT", parent_params)
    except Exception:
        pass

    try:
        node_runner.info(f"multiplicity: {qm_input.multiplicity} optimize: {qm_input.optimization}")
        node_runner.info(f"charge: {qm_input.charge} states: {qm_input.states}")
        node_runner.info(
            f"active_electrons: {qm_input.active_electrons} active_orbitals: {qm_input.active_orbitals}"
        )
        node_runner.info(f"solvent: {qm_input.solvent}")

        # Same Hyperpol-related safety overrides as in ``orca``: ensure that
        # dipole is requested when Hyperpol is on, and upgrade SCF accuracy
        # to at least Tight.
        try:
            elprop_cfg = getattr(qm_input, "elprop", None)
            if elprop_cfg is not None and getattr(elprop_cfg, "Hyperpol", False):
                if not getattr(elprop_cfg, "Dipole", False):
                    node_runner.info(
                        "ElProp Hyperpol requested but Dipole=False; overriding elprop.Dipole -> True "
                        "to ensure dipole moment is available for hyperpolarizability alignment."
                    )
                    elprop_cfg.Dipole = True

                if qm_input.scf_accuracy in {
                    SCFAccuracy.Sloppy,
                    SCFAccuracy.Loose,
                    SCFAccuracy.Medium,
                    SCFAccuracy.Strong,
                }:
                    old_acc = qm_input.scf_accuracy
                    qm_input.scf_accuracy = SCFAccuracy.Tight
                    node_runner.info(
                        f"ElProp Hyperpol requested with scf_accuracy={old_acc.value}; "
                        "overriding to 'Tight' for more reliable hyperpolarizability."
                    )
        except Exception as e_over:
            node_runner.warning(f"Failed to apply Hyperpol-dependent overrides: {e_over}")
        blocks = set_orca_memory_and_pal_options_according_to_slurm_parameters(
            node_runner,
            blocks=None,
            kwargs=kwargs,
            node_slurm_params=orca_jinja._node_parameters.slurm_parameters,
        )

        if qm_input.active_electrons > 0:  # CASSCF
            first_line, cas_scf_spec_bloc, cas_scf_output_block = make_casscf_block(qm_input)
            blocks.append(cas_scf_spec_bloc)
            blocks.append(cas_scf_output_block)
        else:
            aux_basis = (
                qm_input.basis_set.aux_basis.aux_basis.value
                if qm_input.basis_set.aux_basis.aux_basis != AuxBasisEnum.NONE
                else ""
            )
            first_line = set_method_and_basis_set_for_non_casscf_methods(
                node_runner,
                qm_input,
                aux_basis,
                first_line="",
            )
            if qm_input.multiplicity > 1 or qm_input.open_shell_calculation:
                first_line += " UHF"

            if qm_input.optimization:
                opt_acc_mapping = {
                    OptimizationAccuracy.Sloppy:    "LooseOpt",
                    OptimizationAccuracy.Loose:     "LooseOpt",
                    OptimizationAccuracy.Medium:    "Opt",
                    OptimizationAccuracy.Strong:    "TightOpt",
                    OptimizationAccuracy.Tight:     "TightOpt",
                    OptimizationAccuracy.VeryTight: "VeryTightOpt",
                    OptimizationAccuracy.Extreme:   "VeryTightOpt",
                }

                opt_acc = getattr(qm_input, "optimization_accuracy", OptimizationAccuracy.Medium)
                options_to_log = {
                    OptimizationAccuracy.Sloppy,
                    OptimizationAccuracy.Strong,
                    OptimizationAccuracy.Extreme,
                }

                keyword = opt_acc_mapping.get(opt_acc, "Opt")

                if opt_acc in options_to_log:
                    node_runner.info(
                        f"Optimization accuracy {opt_acc.value} mapped to ORCA keyword {keyword}"
                    )

                first_line += f" {keyword}"

                geom_block = f"%geom\n MaxIter {qm_input.max_optimization_iterations}\nend\n\n"
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
            c["index"] + 1 for c in constraints if c.get("type") == "frozen" and "index" in c
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
                first_line, blocks = use_gbw_restart(node_runner, qm_input, first_line, blocks)

        if qm_input.first_line:
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
        node_runner.info_files.append(
            FileStack.from_local_file("orca.inp", in_memory=True, is_hashable=True, secure_source=True)
        )
        node_runner.info("input files done")
    except Exception as e:
        return node_runner.fail(f"Error generating input files {str(e)}")

    # parent_parameters.resource may be a Resource model or a plain string;
    # normalise to a simple string for command construction.
    resource_obj = kwargs["parent_parameters"].resource
    resource = getattr(resource_obj, "value", str(resource_obj))

    # Try to obtain a command from config.toml first. If that fails, use the
    # same hard-coded fallbacks as the legacy ``orca`` node.
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
        if not qm_input.tolerate_failure:
            if os.path.exists("orca.out"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca.out", in_memory=True, is_hashable=True, secure_source=True
                    )
                )
            if os.path.exists("orca_run.log"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca_run.log", in_memory=True, is_hashable=True, secure_source=True
                    )
                )
            return node_runner.fail("Error running ORCA calculation")

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
            orca_run = OrcaRun(path_without_extension)
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
            for molecule in orca_result.structures.molecules:
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
        # directly from the OrcaRun instance.
        try:
            elprop_result = QMResult_elprop.from_orca_output(
                orca_run,
                parent_qm_result=orca_result,
                task_id=node_runner.task_id,
            )
            node_runner.orca_elprop_result = elprop_result
            node_runner.info("QMResult_elprop created from OrcaRun (orca_jinja node)")

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
            node_runner.warning(f"Failed to construct QMResult_elprop from OrcaRun: {e_elprop}")

        node_runner.orca_result = orca_result
        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error reading ORCA result: {str(e)}")



@node(parameters=parameters)
async def orca_jinja_no_restart_from_gbw(qm_input: QMInput, **kwargs) -> SimstackResult:
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
        **kwargs: Additional arbitrary keyword arguments used for
            configuration or overriding specific functionalities. These
            typically include ``parameters`` / ``parent_parameters``
            describing runtime resources (e.g. Slurm configuration).

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

    from .orca_main_lib import (
        add_grid_to_simple_input_line,
        add_tddft_block_if_needed,
        add_electronic_properties_block,
        make_casscf_block,
        set_orca_memory_and_pal_options_according_to_slurm_parameters,
        set_method_and_basis_set_for_non_casscf_methods,
        set_orca_optimization_options
    )
    node_runner = NodeRunner("orca_jinja", logger=logger, **kwargs)

    # Log effective Slurm parameters actually seen by the ORCA node.
    try:
        local_params = kwargs.get("parameters", None)
        parent_params = kwargs.get("parent_parameters", None)
        if local_params is None:
            try:
                local_params = orca_jinja._node_parameters
            except AttributeError:
                local_params = None

        def _log_slurm_params(label: str, params_obj) -> None:
            if params_obj is None:
                node_runner.info(f"[SLURM] {label}: parameters=None")
                return
            slurm = getattr(params_obj, "slurm_parameters", None)
            node_runner.info(
                "[SLURM] %s: resource=%s, queue=%s, nodes=%s, tasks_per_node=%s, cpus_per_task=%s, mem=%s, time=%s"
                % (
                    label,
                    getattr(params_obj, "resource", None),
                    getattr(params_obj, "queue", None),
                    getattr(slurm, "nodes", None) if slurm else None,
                    getattr(slurm, "tasks_per_node", None) if slurm else None,
                    getattr(slurm, "cpus_per_task", None) if slurm else None,
                    getattr(slurm, "mem", None) if slurm else None,
                    getattr(slurm, "time", None) if slurm else None,
                )
            )

        _log_slurm_params("EFFECTIVE", local_params)
        _log_slurm_params("PARENT", parent_params)
    except Exception:
        pass

    try:
        node_runner.info(f"multiplicity: {qm_input.multiplicity} optimize: {qm_input.optimization}")
        node_runner.info(f"charge: {qm_input.charge} states: {qm_input.states}")
        node_runner.info(
            f"active_electrons: {qm_input.active_electrons} active_orbitals: {qm_input.active_orbitals}"
        )
        node_runner.info(f"solvent: {qm_input.solvent}")

        # Same Hyperpol-related safety overrides as in ``orca``: ensure that
        # dipole is requested when Hyperpol is on, and upgrade SCF accuracy
        # to at least Tight.
        try:
            elprop_cfg = getattr(qm_input, "elprop", None)
            if elprop_cfg is not None and getattr(elprop_cfg, "Hyperpol", False):
                if not getattr(elprop_cfg, "Dipole", False):
                    node_runner.info(
                        "ElProp Hyperpol requested but Dipole=False; overriding elprop.Dipole -> True "
                        "to ensure dipole moment is available for hyperpolarizability alignment."
                    )
                    elprop_cfg.Dipole = True

                if qm_input.scf_accuracy in {
                    SCFAccuracy.Sloppy,
                    SCFAccuracy.Loose,
                    SCFAccuracy.Medium,
                    SCFAccuracy.Strong,
                }:
                    old_acc = qm_input.scf_accuracy
                    qm_input.scf_accuracy = SCFAccuracy.Tight
                    node_runner.info(
                        f"ElProp Hyperpol requested with scf_accuracy={old_acc.value}; "
                        "overriding to 'Tight' for more reliable hyperpolarizability."
                    )
        except Exception as e_over:
            node_runner.warning(f"Failed to apply Hyperpol-dependent overrides: {e_over}")
        blocks = set_orca_memory_and_pal_options_according_to_slurm_parameters(
            node_runner,
            blocks=None,
            kwargs=kwargs,
            node_slurm_params=orca_jinja._node_parameters.slurm_parameters,
        )

        if qm_input.active_electrons > 0:  # CASSCF
            first_line, cas_scf_spec_bloc, cas_scf_output_block = make_casscf_block(qm_input)
            blocks.append(cas_scf_spec_bloc)
            blocks.append(cas_scf_output_block)
        else:
            aux_basis = (
                qm_input.basis_set.aux_basis.aux_basis.value
                if qm_input.basis_set.aux_basis.aux_basis != AuxBasisEnum.NONE
                else ""
            )
            first_line = set_method_and_basis_set_for_non_casscf_methods(
                node_runner,
                qm_input,
                aux_basis,
                first_line="",
            )
            if qm_input.multiplicity > 1 or qm_input.open_shell_calculation:
                first_line += " UHF"

            first_line, blocks = set_orca_optimization_options(node_runner, qm_input, first_line, blocks)
            if not getattr(qm_input, "optimization", False) and qm_input.gradients:
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
            c["index"] + 1 for c in constraints if c.get("type") == "frozen" and "index" in c
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
        #no restart for this one- can also later be switched with a flag- depending on whether that should be done with a qm input functionality or with a node flag
        #if os.path.exists("orca.gbw"):
        #    node_runner.info("found orca.gbw file. Using it for geometry optimization. ")
        #    first_line += " MORead"
        #    blocks.append("%moinp \"orca.gbw\"\n\n")
        #else:
        #    if qm_input.restart_files is not None and len(qm_input.restart_files) > 0:
        #        first_line, blocks = use_gbw_restart(node_runner, qm_input, first_line, blocks)

        if qm_input.first_line:
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
        node_runner.info_files.append(
            FileStack.from_local_file("orca.inp", in_memory=True, is_hashable=True, secure_source=True)
        )
        node_runner.info("input files done")
    except Exception as e:
        return node_runner.fail(f"Error generating input files {str(e)}")

    # parent_parameters.resource may be a Resource model or a plain string;
    # normalise to a simple string for command construction.
    resource_obj = kwargs["parent_parameters"].resource
    resource = getattr(resource_obj, "value", str(resource_obj))

    # Try to obtain a command from config.toml first. If that fails, use the
    # same hard-coded fallbacks as the legacy ``orca`` node.
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
        if not qm_input.tolerate_failure:
            if os.path.exists("orca.out"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca.out", in_memory=True, is_hashable=True, secure_source=True
                    )
                )
            if os.path.exists("orca_run.log"):
                node_runner.info_files.append(
                    FileStack.from_local_file(
                        "orca_run.log", in_memory=True, is_hashable=True, secure_source=True
                    )
                )
            return node_runner.fail("Error running ORCA calculation")

    orca_run = None
    orca_out_contents = None
    try:
        if os.path.exists("orca.out"):
            # Cache the ORCA output right after the run for later
            # post-processing, avoiding a second disk access. Perform this
            # before any FileStack interaction with the path.
            with open("orca.out", "r", encoding="utf-8", errors="ignore") as f:
                orca_out_contents = f.read()

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
            for molecule in orca_result.structures.molecules:
                molecule.smiles = qm_input.molecule.smiles
                molecule.formula = qm_input.molecule.formula

            node_runner.info("done standard ORCA parsing")

            # Use cached ORCA output contents captured right after the run.
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
                    node_runner.info("Orbital energies parsed and set on QMResult (orca_jinja_no_restart_from_gbw node)")
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
        # directly from the OrcaRun instance.
        try:
            elprop_result = QMResult_elprop.from_orca_output(
                orca_run,
                parent_qm_result=orca_result,
                task_id=node_runner.task_id,
            )
            node_runner.orca_elprop_result = elprop_result
            node_runner.info("QMResult_elprop created from OrcaRun (orca_jinja node)")

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
            node_runner.warning(f"Failed to construct QMResult_elprop from OrcaRun: {e_elprop}")

        node_runner.orca_result = orca_result
        return node_runner.succeed()
    except Exception as e:
        return node_runner.fail(f"Error reading ORCA result: {str(e)}")




###old stuff defined by Prof.W. - should be removed, but other project files reference this
def make_water() -> Molecule:
    """Create a water molecule with 2 hydrogen and 1 oxygen."""

    coords = [
        [0.0, 0.0, 0.05],  # O atom at origin
        [0.0, 0.757, 0.986],  # H atom 1
        [0.0, -0.757, 0.586],  # H atom 2
    ]

    species = ["O", "H", "H"]
    molecule = Molecule()
    for element, coord in zip(species, coords):
        atom = Atom.from_coords(element=element, coords=coord)
        molecule.add_atom(atom)

    return molecule


async def water_tddft():
    """Create a water molecule and run a TDDFT calculation on it."""

    context.initialize()
    molecule = make_water()
    # Create a QMInput object for the TDDFT calculation
    orca_input = QMInput(molecule=molecule, basis_set=BasisSet(basis_set="def2-SVP", aux_basis=AuxBasis(aux_basis="def2/J")),
                         functional=Functional(functional="B3LYP", dispersion_correction=DispersionCorrection()),
                         states=4, excited_states=True, gradients=False)
    # Run the ORCA calculation
    result = await orca(orca_input)
    return result


async def water():
    """
    Create a water molecule and run a TDDFT calculation on it
    """
    await context.initialize()
    molecule = make_water()
    # Create a QMInput object for the TDDFT calculation
    orca_input = QMInput(molecule=molecule, basis_set=BasisSet(basis_set="def2-SVP", aux_basis=AuxBasis(aux_basis="none")),
                         functional=Functional(functional="B3LYP", dispersion_correction=DispersionCorrection()),
                         states=1, excited_states=True, optimization=True)
    # Run the ORCA calculation
    result = await orca(orca_input)
    return result


async def ethylene_casscf():
    """Create an ethylene molecule and run a CASSCF calculation on it."""

    context.initialize()
    coords = [
        [0.0, 0.0, 0.0],  # C1
        [1.34, 0.0, 0.0],  # C2
        [0.67, 1.16, 0.0],  # H1
        [1.67, 1.16, 0.0],  # H2
        [0.67, -1.16, 0.0],  # H3
        [1.67, -1.16, 0.0],  # H4
    ]
    species = ["C", "C", "H", "H", "H", "H"]
    molecule = Molecule()
    for element, coord in zip(species, coords):
        atom = Atom.from_coords(element=element, coords=coord)
        molecule.add_atom(atom)

    orca_input = QMInput(molecule=molecule,
                         basis_set=BasisSet(basis_set=BasisSetEnum.cc_pVDZ),
                         functional=Functional(functional="B3LYP", dispersion_correction=DispersionCorrection()),
                         active_electrons=4,
                         active_orbitals=4,
                         states=4,
                         excited_states=True,
                         gradients=False)
    result = await orca(orca_input)
    return result






if __name__ == "__main__":
    # Simple manual test hook; delegates to the example helper defined in
    # examples/testing/wenzel_examples_for_orca.py
    import asyncio
    from examples.testing.wenzel_examples_for_orca import water

    asyncio.run(water())
