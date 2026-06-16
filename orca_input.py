import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

from applications.electronic_structure import MoleculeList, Molecule
from molecular_qm_models.qm_input import QMInput, QMMethod, OptimizationAccuracy, SCFAccuracy
from molecular_qm_models.auxiliary_basis import AuxBasisEnum
from simstack.models.parameters import SlurmParameters


def orca_input_factory(qm_input: QMInput, **kwargs) -> "OrcaInput":
    """
    Returns the appropriate OrcaInput subclass based on the provided QMInput.
    """
    if qm_input.active_electrons > 0 or qm_input.method == QMMethod.CASSCF:
        return OrcaInputCASSCF(qm_input, **kwargs)
    elif qm_input.method == QMMethod.DFT:
        return OrcaInputDFT(qm_input, **kwargs)
    elif qm_input.method == QMMethod.TDDFT:
        return OrcaInputTDDFT(qm_input, **kwargs)
    else:
        return OrcaInputSCF(qm_input, **kwargs)


class OrcaInput(ABC):
    """
    Abstract base class for ORCA input generation.
    """

    def __init__(self, qm_input: QMInput, **kwargs):
        self.qm_input = qm_input
        self.kwargs = kwargs
        self.node_runner = kwargs.get("node_runner")
        self._first_line = ""
        self._blocks: List[str] = []
        
    async def molecular_input_block(self):
        """
        Generates the molecular input block for quantum mechanics (QM) calculations.

        This method constructs and appends appropriate molecular input data to the
        internal blocklist required for QM computations. It handles restart scenarios
        by checking for specific file artifacts and incorporates geometry optimization
        data when applicable. If restart files are found, it modifies the blocks and
        flags to use them for input geometry; otherwise, it falls back to generating
        input geometry based on the molecule's atomic information.

        Raises:
            Exception: If there is an issue accessing or using restart files during
                       the creation of molecular input blocks.

        """

        use_input_geometry = True
        qm_input = self.qm_input
        node_runner = self.node_runner
        if qm_input.restartable:
            if qm_input.restart_files is not None and len(qm_input.restart_files) > 0:
                gbw_file_name = "orca.gbw"
                orca_gbw = await qm_input.restart_files.find(gbw_file_name)
                if orca_gbw is not None:
                    orca_gbw.get(Path.cwd())
                    self._first_line += " MORead"
                    self._blocks.append(f"%moinp \"{gbw_file_name}\"\n\n")
                    node_runner.info(f"found {gbw_file_name} restart file. Using it for geometry optimization. ")
                coordinate_file_name = "orca.xyz"
                orca_xyz = await qm_input.restart_files.find(coordinate_file_name)
                if orca_xyz is not None:
                    orca_xyz.get(Path.cwd())
                    use_input_geometry = False
                    self._blocks.append(f"* xyzfile {qm_input.charge} {qm_input.multiplicity} {coordinate_file_name}\n")
                    node_runner.info(f"found {coordinate_file_name} restart file. Using it for geometry optimization. ")
            else:
                if qm_input.optimization and os.path.exists("orca_traj.xyz"):
                    molecule_list = MoleculeList.from_xyz(qm_input.molecule, "orca_traj.xyz")
                    if len(molecule_list) > 0:
                        molecule = await molecule_list.get_molecule(len(molecule_list) - 1)
                        self._generate_xyz_block(molecule)
                    self.node_runner.info("found orca.opt file. Using it for geometry optimization. ")
                    use_input_geometry = False
                    self._blocks.append(f"* xyzfile {qm_input.charge} {qm_input.multiplicity} orca.opt\n")
                elif os.path.exists("orca.xyz"):
                    self.node_runner.info("found orca.xyz file. Using it for geometry optimization. ")
                    use_input_geometry = False
                    self._blocks.append(f"* xyzfile {qm_input.charge} {qm_input.multiplicity} orca.xyz\n")
        
                if os.path.exists("orca.gbw"):
                    self.node_runner.info("found orca.gbw file. Using it for geometry optimization. ")
                    if " MORead" not in self.qm_input.first_line:
                        self._first_line += " MORead"
                    self._blocks.append("%moinp \"orca.gbw\"\n\n")
            
        if use_input_geometry:
            await self._generate_xyz_block(qm_input.molecule)

    def _generate_xyz_block(self, molecule: Molecule):
        geometry_block = f"* xyz {self.qm_input.charge} {self.qm_input.multiplicity}\n"
        for atom in self.qm_input.molecule.atoms:
            geometry_block += f"  {atom.element:<2} {atom.x:>12.6f} {atom.y:>12.6f} {atom.z:>12.6f}\n"
        geometry_block += "*\n"
        self._blocks.append(geometry_block)

    @property
    def first_line(self) -> str:
        """Returns the first line of the ORCA input file."""
        if self.qm_input.first_line:
            first_line = f" {self.qm_input.first_line}"
        else:
            first_line = self._get_base_first_line() + self._first_line + "\n\n"
        return first_line

    @abstractmethod
    def _get_base_first_line(self) -> str:
        """Returns the method-specific base first line."""
        pass

    @abstractmethod
    def safety_overrides(self) -> str:
        """Returns the safety overrides for the calculation."""
        pass

    @property
    def blocks(self) -> List[str]:
        """Returns a list of blocks for the ORCA input file."""
        self.molecular_input_block() # this generates one or more blocks for molecular input
        blocks = self._get_base_blocks()
        blocks = self._add_constraint_blocks(blocks)

        blocks.extend(self._blocks)
        return blocks

    @abstractmethod
    def _get_base_blocks(self) -> List[str]:
        """Returns the method-specific base blocks."""
        pass

    def _add_constraint_blocks(self, blocks: List[str]) -> List[str]:
        constraints = self.qm_input.molecule.properties.get("constraints", [])
        # TODO constraints must be implemented
        if self.node_runner:
            self.node_runner.info(f"constraints: {constraints}")
        frozen_atoms = [
            c["index"] + 1 for c in constraints if c.get("type") == "frozen" and "index" in c
        ]
        if frozen_atoms:
            frozen_atoms_block = "%geom\n Constraints\n"
            for atom_idx in frozen_atoms:
                frozen_atoms_block += f"  {{ C {atom_idx} C }}\n"
            frozen_atoms_block += "end\nend\n\n"
            blocks.append(frozen_atoms_block)
        return blocks

    @property
    def slurm_params(self) -> Optional[SlurmParameters]:
        """Returns the effective SlurmParameters for the calculation."""
      
        parent_params = self.kwargs.get("parent_parameters")
        if parent_params is not None and getattr(parent_params, "slurm_parameters", None) is not None:
            return parent_params.slurm_parameters
        return SlurmParameters()

    @property
    def cores_block(self) -> str:
        """Returns the %pal block based on Slurm parameters."""
        slurm_params = self.slurm_params
        if slurm_params is None:
            return ""
        if (
            slurm_params.cpus_per_task is not None
            and slurm_params.cpus_per_task > 0
            and slurm_params.tasks_per_node is not None
            and slurm_params.tasks_per_node > 0
        ):
            nprocs = int(slurm_params.cpus_per_task) * int(slurm_params.tasks_per_node)
            if nprocs > 1:
                return f"%pal\n nprocs {nprocs}\nend\n\n"
        elif (
            slurm_params.cpus_per_task is not None
            and slurm_params.cpus_per_task > 1
        ):
            return f"%pal\n nprocs {int(slurm_params.cpus_per_task)}\nend\n\n"
        elif (
            slurm_params.tasks_per_node is not None
            and slurm_params.tasks_per_node > 1
        ):
            return f"%pal\n nprocs {int(slurm_params.tasks_per_node)}\nend\n\n"
        return ""

    @property
    def memory_block(self) -> str:
        """Returns the %maxcore block based on Slurm parameters."""
        slurm_params = self.slurm_params
        if slurm_params is None or slurm_params.mem is None or slurm_params.mem == "":
            return ""

        # Derive effective CPU count to calculate memory per core
        if (
            slurm_params.cpus_per_task is not None
            and slurm_params.cpus_per_task > 0
            and slurm_params.tasks_per_node is not None
            and slurm_params.tasks_per_node > 0
        ):
            n_cpus_eff = int(slurm_params.cpus_per_task) * int(slurm_params.tasks_per_node)
        elif slurm_params.cpus_per_task is not None and slurm_params.cpus_per_task > 0:
            n_cpus_eff = int(slurm_params.cpus_per_task)
        elif slurm_params.tasks_per_node is not None and slurm_params.tasks_per_node > 0:
            n_cpus_eff = int(slurm_params.tasks_per_node)
        else:
            n_cpus_eff = 1

        try:
            if "G" in slurm_params.mem:
                total_memory = int(slurm_params.mem.replace("G", "").strip()) * 1024
                memory_per_cpu = int(float(total_memory) / n_cpus_eff) + 100
                return f"%maxcore {memory_per_cpu}\n"
            elif "M" in slurm_params.mem:
                memory_per_cpu = int(
                    float(slurm_params.mem.replace("M", "").strip()) / n_cpus_eff
                ) + 100
                return f"%maxcore {memory_per_cpu}\n"
        except (ValueError, ZeroDivisionError):
            pass

        return ""

    @property
    def parameters_block(self) -> str:
        """Returns the combined memory and cores parameters block."""
        return self.cores_block + self.memory_block


class OrcaInputSCF(OrcaInput):

    def safety_overrides(self) -> str:
        # Same Hyperpolarozibility-related safety overrides as in ``orca``: ensure that
        # dipole is requested when Hyperpol is on, and upgrade SCF accuracy
        # to at least Tight.
        try:
            elprop_cfg = getattr(self.qm_input, "elprop", None)
            if elprop_cfg is not None and getattr(elprop_cfg, "Hyperpol", False):
                if not getattr(elprop_cfg, "Dipole", False):
                    self.node_runner.info(
                        "ElProp Hyperpol requested but Dipole=False; overriding elprop.Dipole -> True "
                        "to ensure dipole moment is available for hyperpolarizability alignment."
                    )
                    elprop_cfg.Dipole = True

                if self.qm_input.scf_accuracy in {
                    SCFAccuracy.Sloppy,
                    SCFAccuracy.Loose,
                    SCFAccuracy.Medium,
                    SCFAccuracy.Strong,
                }:
                    old_acc = self.qm_input.scf_accuracy
                    self.qm_input.scf_accuracy = SCFAccuracy.Tight
                    self.node_runner.info(
                        f"ElProp Hyperpol requested with scf_accuracy={old_acc.value}; "
                        "overriding to 'Tight' for more reliable hyperpolarizability."
                    )
        except Exception as e_over:
            self.node_runner.warning(f"Failed to apply Hyperpol-dependent overrides: {e_over}")


    def _get_base_first_line(self) -> str:
        from applications.electronic_structure.orca.lib.orca_main_lib import (
            add_grid_to_simple_input_line,
            set_method_and_basis_set_for_non_casscf_methods,
        )
        aux_basis = (
            self.qm_input.basis_set.aux_basis.aux_basis.value
            if self.qm_input.basis_set.aux_basis.aux_basis != AuxBasisEnum.NONE
            else ""
        )
        first_line = set_method_and_basis_set_for_non_casscf_methods(
            self.node_runner,
            self.qm_input,
            aux_basis,
            first_line="",
        )
        if self.qm_input.multiplicity > 1 or self.qm_input.open_shell_calculation:
            first_line += " UHF"

        first_line = self._add_standard_keywords(first_line)
        first_line = add_grid_to_simple_input_line(first_line, self.qm_input)
        return first_line

    def _add_standard_keywords(self, first_line: str) -> str:
        if self.qm_input.optimization:
            opt_acc_mapping = {
                OptimizationAccuracy.Sloppy: "LooseOpt",
                OptimizationAccuracy.Loose: "LooseOpt",
                OptimizationAccuracy.Medium: "Opt",
                OptimizationAccuracy.Strong: "TightOpt",
                OptimizationAccuracy.Tight: "TightOpt",
                OptimizationAccuracy.VeryTight: "VeryTightOpt",
                OptimizationAccuracy.Extreme: "VeryTightOpt",
            }
            opt_acc = getattr(self.qm_input, "optimization_accuracy", OptimizationAccuracy.Medium)
            keyword = opt_acc_mapping.get(opt_acc, "Opt")

            options_to_log = {
                OptimizationAccuracy.Sloppy,
                OptimizationAccuracy.Strong,
                OptimizationAccuracy.Extreme,
            }
            if self.node_runner and opt_acc in options_to_log:
                self.node_runner.info(f"Optimization accuracy {opt_acc.value} mapped to ORCA keyword {keyword}")

            first_line += f" {keyword}"
        elif self.qm_input.gradients:
            first_line += " ENGRAD"

        if self.node_runner:
            self.node_runner.info(f"frequencies: {self.qm_input.frequencies}")
        if self.qm_input.frequencies:
            first_line += " FREQ"

        if self.qm_input.solvent.lower() != "none":
            first_line += f" CPCM({self.qm_input.solvent.upper()})"

        return first_line

    def _get_base_blocks(self) -> List[str]:
        from applications.electronic_structure.orca.lib.orca_main_lib import (
            add_tddft_block_if_needed,
            add_electronic_properties_block,
        )
        blocks = [self.parameters_block]

        if self.qm_input.optimization:
            geom_block = f"%geom\n MaxIter {self.qm_input.max_optimization_iterations}\nend\n\n"
            blocks.append(geom_block)

        blocks = add_electronic_properties_block(self.qm_input, blocks)

        conv_keyword = self.qm_input.scf_accuracy.value
        scf_block = (
            f"%scf\n Convergence {conv_keyword}\n MaxIter {self.qm_input.max_scf_iterations}\nend\n\n"
        )
        blocks.append(scf_block)

        blocks = add_tddft_block_if_needed(self.qm_input, blocks)
        return blocks


class OrcaInputDFT(OrcaInputSCF):
    pass


class OrcaInputTDDFT(OrcaInputDFT):
    pass


class OrcaInputCASSCF(OrcaInput):

    def safety_overrides(self) -> str:
        # TODO: Implement safety overrides for CASSCF
        return ""

    def _get_base_first_line(self) -> str:
        return self.qm_input.basis_set.basis_set + " CASSCF TightSCF"

    def _get_base_blocks(self) -> List[str]:
        qm_input = self.qm_input
        self._blocks.append("%casscf\n" +
                            f" nel {qm_input.active_electrons}\n" +
                            f" norb {qm_input.active_orbitals}\n" +
                            f" nroots {qm_input.states}\n" +
                            f" printlevel {qm_input.print_level}\n" +
                            " DoNTO true             # transition orbitals \n" +
                            " PrintWF 1               # Print wavefunction details\n" +
                            " maxiter 100\n" +
                            "end\n\n")
        # blocks.append(cas_scf_block)
        self._blocks.append("%output\n" + \
                            "  print[p_loewdin] 2      # Löwdin population analysis (detailed)\n" + \
                            "  print[p_mulliken] 2     # Mulliken for comparison\n" + \
                            "  print[p_hirshfeld] 1    # Hirshfeld analysis\n" + \
                            "  print[p_orbpopmo_l] 1   # Löwdin orbital populations per MO\n" + \
                            "  # Orbital printing\n" + \
                            "  printLevel 4            # Maximum output detail\n" + \
                            "  print[p_basis] 2        # Basis set information\n" + \
                            "  print[p_mos] 1          # Molecular orbital information\n" + \
                            "  print[p_mayer] 1        # Mayer Bond Orders\n" + \
                            "end\n\n")
        return self._blocks
