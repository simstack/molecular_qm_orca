"""Build :class:`QMResult` instances from parsed ORCA output objects."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING

from molecular_qm_models.molecule import Atom, Molecule, MoleculeList
from molecular_qm_models.qm_result import QMResult
from simstack.core.context import context
from simstack.models.file_list import FileList
from simstack.models.files import FileStack

if TYPE_CHECKING:
    from molecular_qm_orca.lib.orca_output import OrcaOutput

logger = logging.getLogger("make_qm_result")

# Artifacts that belong on QMResult.files (not node info_files).
ORCA_QMRESULT_FILES = (
    "orca.gbw",
    "orca.xyz",
    "orca.densities",
    "orca.densitiesinfo",
    "orca.opt",
    "orca.engrad",
    "orca_trj.xyz",
    "orca.hess",
    "orca.property.txt",
    "orca.DFTMRCI.inp",
    "orca.bkji",
)

# Scalar / list fields copied from OrcaOutput (or legacy OrcaRun) onto QMResult.
_QMRESULT_ATTRS = (
    "charge",
    "dipole",
    "final_energy",
    "dipole_moment",
    "energies",
    "scf_energies",
    "scf_converged",
    "optimization_converged",
    "normal_termination",
    "status",
    "error",
)


def _molecule_from_structure(structure: Any) -> Molecule:
    """Convert a parsed geometry to :class:`Molecule`.

    Accepts either our :class:`Molecule` (``OrcaOutput``) or a pymatgen-like
    object with ``.sites`` (legacy ``OrcaRun``).
    """
    if structure is None:
        raise ValueError("structure is None")
    if isinstance(structure, Molecule):
        return Molecule.from_atoms(
            list(structure.atoms),
            properties=dict(structure.properties) if structure.properties else None,
        )
    molecule = Molecule()
    for site in structure.sites:
        element = getattr(site, "label", None) or getattr(site, "species_string", None)
        atom = Atom.from_coords(element=str(element), coords=list(site.coords))
        molecule.add_atom(atom)
    return molecule


def _copy_scalar_fields(orca_run: Any) -> dict[str, Any]:
    """Map convergence / energy / dipole fields from a parsed ORCA object."""
    result: dict[str, Any] = {}
    for attr in _QMRESULT_ATTRS:
        if hasattr(orca_run, attr):
            if attr == "error":
                continue
            result[attr] = getattr(orca_run, attr)

        #the following does not work due to the error reqiring an msg argument
        #  if callable(getattr(orca_run, attr)):
        #         result[attr] = getattr(orca_run, attr)()
        #     else:
        #         result[attr] = getattr(orca_run, attr)


    # Legacy OrcaRun used the misspelled ``scf_converge``.
    if "scf_converged" not in result or result["scf_converged"] is None:
        legacy = getattr(orca_run, "scf_converge", None)
        if legacy is not None:
            result["scf_converged"] = bool(legacy)

    if "optimization_converged" not in result:
        result["optimization_converged"] = False

    return result


async def _collect_qmresult_files() -> FileList:
    """Attach existing ORCA artifact files into a :class:`FileList`."""
    file_list = FileList()
    seen: set[str] = set()
    for file_path in ORCA_QMRESULT_FILES:
        if not os.path.exists(file_path) or file_path in seen:
            continue
        try:
            file_stack = FileStack.from_local_file(
                file_path, in_memory=False, is_hashable=True, secure_source=True
            )
            await context.db.save(file_stack)
            file_list.append(file_stack)
            seen.add(file_path)
        except Exception as e:
            logger.warning("Failed to attach ORCA result file %s: %s", file_path, e)
    return file_list


async def from_orca_output(
    orca_run: "OrcaOutput",
    task_id: Optional[str] = None,
) -> QMResult:
    """Create a :class:`QMResult` from an ``OrcaOutput`` or legacy ``OrcaRun``.

    Populates scalar convergence/energy fields from the parsed object and
    collects restart/geometry artifacts listed in :data:`ORCA_QMRESULT_FILES`
    into ``QMResult.files``.
    """
    logger.info("Reading results from run task_id: %s", task_id)

    result_dict = _copy_scalar_fields(orca_run)

    try:
        final_structure = _molecule_from_structure(orca_run.final_structure)
        final_structure = await context.db.save(final_structure)
        logger.info("Reading results from run task_id: %s --- Final structure done", task_id)
    except Exception as e:
        logger.info(
            "Reading results from run task_id: %s --- Final structure FAILED: %s",
            task_id,
            e,
        )
        final_structure = Molecule()

    molecule_list = MoleculeList()
    try:
        for structure in orca_run.structures:
            molecule = _molecule_from_structure(structure)
            molecule = await context.db.save(molecule)
            molecule_list.add_molecule(molecule)
        await context.db.save(molecule_list)
        logger.info("Reading results from run task_id: %s --- Structures done", task_id)
    except Exception as e:
        logger.info(
            "Reading results from run task_id: %s --- Structures FAILED: %s",
            task_id,
            e,
        )

    file_list = await _collect_qmresult_files()
    logger.info("Reading results from run task_id: %s --- FileList Done", task_id)

    result_dict["structures"] = molecule_list
    result_dict["final_structure"] = final_structure
    result_dict["files"] = file_list

    result = QMResult(**result_dict)
    logger.info("Reading results from run task_id: %s --- Result creation done", task_id)
    return result
