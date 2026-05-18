from molecular_qm_models.qm_input import QMMethod, OptimizationAccuracy


def add_grid_to_simple_input_line(first_line, qm_input):
    '''Adds grid specifications to the first line of the ORCA input if a grid type is specified in qm_input.
    to declutter the main moved to lib file - can later be replaced by version-specific generators to be loaded
    Args:                    first_line (str): The initial line of the ORCA input file to which grid specifications will
        be appended.                    qm_input (QMInput): The quantum mechanical input object containing the grid type information.
    Returns:                    str: The modified first line of the ORCA input file with grid specifications appended if applicable.'''

    if qm_input.grid_type is not None:
        if qm_input.grid_type == "Grid5":
            grid_level = 3
        elif qm_input.grid_type == "Grid4":
            grid_level = 3
        elif qm_input.grid_type == "Grid3":
            grid_level = 2
        elif qm_input.grid_type == "Grid2":
            grid_level = 2
        else:
            grid_level = 1


        # In ORCA, inside the %method block, Grid and FinalGrid take integer values (1-7)
        #grid_block = f"%method\n Grid {grid_level}\n FinalGrid {grid_level + 1 if grid_level < 7 else grid_level}\nend\n\n"
        #grid_block = f"! Grid{grid_level} FinalGrid{grid_level + 1 if grid_level < 7 else grid_level}\n\n"
        first_line += f" DEFGRID{grid_level}"
    return first_line

def add_tddft_block_if_needed(qm_input, blocks):
    if qm_input.states > 0:
        tddft_block = "%tddft\n"
        tddft_block += f" nroots {qm_input.states}\n"
        tddft_block += f" iroot {qm_input.focus_state}\n"
        # todo when do we need irootmult?
        #tddft_block += f" irootmult {multiplicity_to_string(qm_input.multiplicity)}\n"
        tddft_block += "end\n\n"
        blocks.append(tddft_block)
    return blocks

def add_electronic_properties_block(
    qm_input,
    blocks,
    elprop_comments: dict = {
        "Dipole": "dipole moment",
        "Quadrupole": "quadrupole moment",
        "Polar": "dipole-dipole polarizability",
        "Hyperpol": "dip/dip/dip hyperpolarizability",
        "PolarVelocity": "polarizability w.r.t. velocity perturbations",
        "PolarDipQuad": "dipole-quadrupole polarizability",
        "PolarQuadQuad": "quadrupole-quadrupole polarizability",
    },
):
    """Construct the ORCA %elprop block from QMInput-style settings.

    The *new* QMInput model exposes electrical‑property toggles as
    top‑level boolean fields (``Dipole``, ``Hyperpol``, ...). Older
    workflows, as well as :class:`DummyQMInput`, may still provide an
    embedded ``elprop`` model.

    To keep the helper compatible with both layouts we:

    * Prefer the new top‑level boolean fields when present.
    * Fall back to an embedded ``elprop`` attribute (with ``model_dump``)
      if no top‑level flags are found.
    """

    # Preferred path: new top‑level flags on QMInput
    top_level_fields = [
        "Dipole",
        "Quadrupole",
        "Polar",
        "Hyperpol",
        "PolarVelocity",
        "PolarDipQuad",
        "PolarQuadQuad",
    ]

    elprop_dict: dict[str, bool] = {}

    for field_name in top_level_fields:
        if hasattr(qm_input, field_name):
            value = getattr(qm_input, field_name)
            # Only treat boolean-like flags as elprop toggles
            if isinstance(value, bool):
                elprop_dict[field_name] = value

    # Backwards‑compatibility: legacy embedded ``elprop`` model
    if not elprop_dict:
        elprop_cfg = getattr(qm_input, "elprop", None)
        raw = None
        if elprop_cfg is not None:
            if hasattr(elprop_cfg, "model_dump"):
                raw = elprop_cfg.model_dump()
            elif isinstance(elprop_cfg, dict):
                raw = elprop_cfg

        if isinstance(raw, dict):
            for key, val in raw.items():
                # Coerce to boolean for safety – non‑bool values are
                # interpreted as simple on/off flags.
                elprop_dict[key] = bool(val)

    # Nothing to do if no elprop information is present
    if not elprop_dict:
        return blocks

    elprop_block = "%elprop\n"
    for prop, calculate in elprop_dict.items():
        comment = elprop_comments.get(prop, "")
        line = f" {prop} {'true' if calculate else 'false'}"
        if comment:
            line += f" # {comment}"
        elprop_block += line + "\n"
    elprop_block += "end\n\n"

    blocks.append(elprop_block)
    return blocks


def set_method_and_basis_set_for_non_casscf_methods(node_runner, qm_input, aux_basis, first_line: str = "") -> str:
    """Build the method/basis part of the ORCA first line for non-CASSCF methods.

    If ``first_line`` is non-empty, the method/basis/aux part is appended so
    callers can prepend additional flags (e.g. print levels) before the
    method specification.
    """

    # Correlated wavefunction methods: use the method keyword directly
    if qm_input.method in {
        QMMethod.DLPNO_CCSD,
        QMMethod.DLPNO_CCSD_T,
        QMMethod.CCSD,
        QMMethod.CCSD_T,
    }:
        method_str = qm_input.method.value
        if first_line:
            first_line = (
                f"{first_line} {method_str} {qm_input.basis_set.basis_set.value} "
                f"{aux_basis}"
            )
        else:
            first_line = (
                f"{method_str} {qm_input.basis_set.basis_set.value} "
                f"{aux_basis}"
            )

    # DFT/TDDFT: use functional name plus basis
    elif qm_input.method in {QMMethod.DFT, QMMethod.TDDFT}:
        first_line = (
            f"{qm_input.functional.functional.value} {qm_input.basis_set.basis_set.value} "
            f"{aux_basis}"
        )
    elif qm_input.method == QMMethod.HF:
        first_line = (
            f"HF {qm_input.basis_set.basis_set.value} "
            f"{aux_basis}"
        )
    # Fallback: default to DFT-style input but log that we fell back here
    else:
        first_line = (
            f"{qm_input.functional.functional.value} {qm_input.basis_set.basis_set.value} "
            f"{aux_basis}"
        )
        if node_runner is not None:
            node_runner.info(
                f"Defaulting to DFT with functional {qm_input.functional.functional.value} "
                f"and basis set {qm_input.basis_set.basis_set.value} for method {qm_input.method}"
            )

    return first_line


def set_orca_memory_and_pal_options_according_to_slurm_parameters(
    node_runner,
    blocks,
    kwargs,
    node_slurm_params,
):
    """Configure ORCA %pal and %maxcore blocks from Slurm parameters.

    Prefer runtime Slurm parameters coming from the parent task (reflecting
    the actual SBATCH header). If those are not available, fall back to the
    node's default Slurm parameters provided via ``node_slurm_params``.

    Parameters
    ----------
    node_runner:
        NodeRunner instance used for logging.
    blocks:
        Existing list of ORCA input blocks (or ``None``), which will be
        extended in-place and also returned.
    kwargs:
        Keyword arguments passed into the node; may contain
        ``parent_parameters`` and ``parameters``.
    node_slurm_params:
        Default SlurmParameters object associated with the node
        (e.g. ``orca._node_parameters.slurm_parameters``).
    """

    node_runner.info("Generating input files for ORCA V6.1.1")
    parameters = kwargs.get("parameters", None)

    # Ensure these are always defined, even if something fails below
    pal_block = ""
    mem_block = ""

    try:
        # Prefer runtime Slurm parameters from the parent task, fall back
        # to the node's default parameters if not available.
        parent_params = kwargs.get("parent_parameters")
        if parent_params is not None and getattr(
            parent_params, "slurm_parameters", None
        ) is not None:
            slurm_params = parent_params.slurm_parameters
        else:
            slurm_params = node_slurm_params

        node_runner.info(
            f"Slurm parameters: time={slurm_params.time}, mem={slurm_params.mem}, "
            f"cpus_per_task={slurm_params.cpus_per_task}, tasks-per-node={slurm_params.tasks_per_node}"
        )

        # Derive effective CPU count and %pal block
        if (
            slurm_params.cpus_per_task is not None
            and slurm_params.cpus_per_task > 0
            and slurm_params.tasks_per_node is not None
            and slurm_params.tasks_per_node > 0
        ):
            pal_block = (
                f"%pal\n nprocs {int(slurm_params.cpus_per_task) * int(slurm_params.tasks_per_node)}\nend\n\n"
            )
            N_cpus_eff = int(slurm_params.cpus_per_task) * int(
                slurm_params.tasks_per_node
            )
        elif (
            slurm_params.cpus_per_task is not None
            and slurm_params.cpus_per_task > 0
        ):
            N_cpus_eff = int(slurm_params.cpus_per_task)
        elif (
            slurm_params.tasks_per_node is not None
            and slurm_params.tasks_per_node > 0
        ):
            N_cpus_eff = int(slurm_params.tasks_per_node)
        else:
            node_runner.warning(
                "Slurm parameter 'cpus_per_task' is not set or invalid, defaulting to 1 CPU."
            )
            N_cpus_eff = 1

        # Derive %maxcore (memory per core) block
        if slurm_params.mem is not None and slurm_params.mem != "":
            if "G" in slurm_params.mem:
                total_memory = int(slurm_params.mem.replace("G", "").strip()) * 1024
                memory_per_cpu = int(float(total_memory) / N_cpus_eff) + 100
                mem_block = f"%maxcore {memory_per_cpu}\n"
            elif "M" in slurm_params.mem:
                memory_per_cpu = int(
                    float(slurm_params.mem.replace("M", "").strip()) / N_cpus_eff
                ) + 100
                mem_block = f"%maxcore {memory_per_cpu}\n"
            else:
                node_runner.warning(
                    "Slurm parameter 'mem' is not in expected format (e.g., '2G'), defaulting to 2000 MB total memory."
                )
                mem_block = "%maxcore 2000\n"
    except Exception as e:
        node_runner.warning(
            f"Could not retrieve Slurm parameters within orca helper: {str(e)}"
        )

    if parameters:
        node_runner.info(f"Using parameters: {parameters}")

    blocks = blocks if blocks is not None else []
    if pal_block:
        blocks.append(pal_block)
    if mem_block:
        blocks.append(mem_block)

    return blocks

def make_casscf_block(qm_input):

    first_line = qm_input.basis_set.basis_set + " CASSCF TightSCF"
    cas_scf_block = ("%casscf\n" +
                    f" nel {qm_input.active_electrons}\n" +
                    f" norb {qm_input.active_orbitals}\n" +
                    f" nroots {qm_input.states}\n" +
                    f" printlevel {qm_input.print_level}\n" +
                    " DoNTO true             # transition orbitals \n" +
                    " PrintWF 1               # Print wavefunction details\n" +
                    " maxiter 100\n" +

                    "end\n\n")
    #blocks.append(cas_scf_block)
    output_block = "%output\n" + \
                "  print[p_loewdin] 2      # Löwdin population analysis (detailed)\n" + \
                "  print[p_mulliken] 2     # Mulliken for comparison\n" + \
                "  print[p_hirshfeld] 1    # Hirshfeld analysis\n" + \
                "  print[p_orbpopmo_l] 1   # Löwdin orbital populations per MO\n" + \
                "  # Orbital printing\n" + \
                "  printLevel 4            # Maximum output detail\n" + \
                "  print[p_basis] 2        # Basis set information\n" + \
                "  print[p_mos] 1          # Molecular orbital information\n" + \
                "  print[p_mayer] 1        # Mayer Bond Orders\n" + \
                "end\n\n"
    #blocks.append(output_block)
    return first_line, cas_scf_block,  output_block


def use_gbw_restart(node_runner, qm_input, first_line, blocks, gbw_file_name:str= "orca.gbw"):
    from pathlib import Path

    node_runner.info("found restart files. Using them for geometry optimization. ")
    orca_gbw = qm_input.restart_files.find(gbw_file_name)
    if orca_gbw is not None:
        orca_gbw.get(Path.cwd())
        first_line += " MORead"
        blocks.append(f"%moinp \"{gbw_file_name}\"\n\n")
    else:
        node_runner.error(f"no {gbw_file_name} file found in restart files. Skipping it. ")
    return first_line, blocks


def set_orca_optimization_options(node_runner, qm_input, first_line: str, blocks):
    """Append ORCA optimisation keyword and %geom block based on QMInput settings.

    This helper maps :class:`OptimizationAccuracy` values onto ORCA's
    optimisation keywords (LooseOpt, Opt, TightOpt, VeryTightOpt) and
    appends the corresponding keyword to ``first_line`` together with a
    ``%geom`` block specifying the maximum number of optimisation
    iterations.

    The mapping is approximate for some levels (e.g. Sloppy → LooseOpt,
    Strong → TightOpt, Extreme → VeryTightOpt); such cases are logged
    explicitly via ``node_runner.info`` if a node runner is provided.

    Parameters
    ----------
    node_runner:
        NodeRunner instance used for logging (may be ``None``).
    qm_input:
        QMInput-like object with ``optimization``, ``optimization_accuracy``
        and ``max_optimization_iterations`` attributes.
    first_line:
        Current ORCA first-line string (method/basis/etc.). The
        optimisation keyword will be appended to this.
    blocks:
        List of ORCA input blocks. The ``%geom`` block will be appended
        here if optimisation is enabled.

    Returns
    -------
    tuple[str, list]
        Updated ``first_line`` and ``blocks``.
    """

    # If optimisation is not requested, leave inputs unchanged.
    if not getattr(qm_input, "optimization", False):
        return first_line, blocks

    opt_acc_mapping = {
        OptimizationAccuracy.Sloppy:    "LooseOpt",     # very weak → LooseOpt
        OptimizationAccuracy.Loose:     "LooseOpt",     # still weak → LooseOpt
        OptimizationAccuracy.Medium:    "Opt",          # default
        OptimizationAccuracy.Strong:    "TightOpt",     # stronger → TightOpt
        OptimizationAccuracy.Tight:     "TightOpt",     # still stronger → TightOpt
        OptimizationAccuracy.VeryTight: "VeryTightOpt", # even stronger
        OptimizationAccuracy.Extreme:   "VeryTightOpt", # map to strongest available
    }

    opt_acc = getattr(qm_input, "optimization_accuracy", OptimizationAccuracy.Medium)
    options_to_log = {
        OptimizationAccuracy.Sloppy,
        OptimizationAccuracy.Strong,
        OptimizationAccuracy.Extreme,
    }

    keyword = opt_acc_mapping.get(opt_acc, "Opt")

    if node_runner is not None and opt_acc in options_to_log:
        node_runner.info(
            f"Optimization accuracy {opt_acc.value} mapped to not fully corresponding ORCA keyword {keyword}"
        )

    first_line += f" {keyword}"

    geom_block = f"%geom\n MaxIter {qm_input.max_optimization_iterations}\nend\n\n"
    blocks.append(geom_block)

    return first_line, blocks