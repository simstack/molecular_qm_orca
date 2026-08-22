import re
import io
import logging
from pathlib import Path
from typing import List, Optional

from molecular_qm_models import QMResult
from molecular_qm_models.molecule import MoleculeList, Molecule
from simstack.core.node_runner import NodeRunner

logger = logging.getLogger(__name__)

SCF_NOT_CONVERGED_MSG = (
    "unfortunately, the SCF has not converged. There may be a way out but we have to stop here"
)


class OrcaOutput:
    """Single source of truth for reading and interpreting ``orca.out``."""

    def __init__(
        self,
        node_runner: NodeRunner,
        filename: Path | str = "orca.out",
        content: Optional[str] = None,
    ):
        self.node_runner = node_runner
        self.filename = Path(filename)

        if content is not None:
            self.content = content
        else:
            with open(self.filename, "r", encoding="utf-8", errors="replace") as f:
                self.content = f.read()

        self.charge: int = 0
        self.dipole: Optional[float] = None
        self.final_energy: Optional[float] = None
        self.dipole_moment: Optional[List[float]] = None
        self.energies: List[float] = []
        self.status: Optional[str] = None
        self.error: Optional[str] = None
        self.normal_termination: bool = False

        self.scf_energies: List[float] = []
        self.scf_converged: Optional[bool] = None
        # True only when ORCA reports geometry-opt convergence; remains False
        # for single-point jobs or failed/incomplete optimizations.
        self.optimization_converged: bool = False
        self.structures: MoleculeList = MoleculeList()
        self.final_structure: Optional[Molecule] = None

        self._parse()

        self.qm_result: QMResult | None = None

    @classmethod
    def load_orca_output(
        cls,
        node_runner: NodeRunner,
        filename: Path | str = "orca.out",
    ) -> Optional["OrcaOutput"]:
        """Return a parsed ``OrcaOutput``, or ``None`` if the file is missing."""
        path = Path(filename)
        if not path.exists():
            return None
        return cls(node_runner, path)

    def log_tail(self, n: int = 100) -> None:
        """Write the last ``n`` lines of ``orca.out`` to the node-runner log."""
        lines = self.content.splitlines()
        tail = lines[-n:] if len(lines) > n else lines
        self.node_runner.log(f"----- last {len(tail)} lines of {self.filename.name} -----")
        for line in tail:
            self.node_runner.log(line)
        self.node_runner.log(f"----- end of {self.filename.name} tail -----")

    def _parse(self):
        """Parse the content and fill in attributes"""
        fout = io.StringIO(self.content)

        dipole_patt = re.compile(
            r"^Total Dipole Moment\s+:\s+([+-]?\d+\.\d+)"
            r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
        )
        energy_patt = re.compile(r"FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)")
        scf_patt = re.compile(
            r"Total Energy\s+:\s+([+|-]?\d+\.\d+) Eh\s+([+|-]?\d+\.\d+) eV"
        )
        cart_patt = re.compile(
            r"^\s+([a-zA-Z]{1,2}|-)\s+([+-]?\d+\.\d+)"
            r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
        )
        scf_cvg_patt = re.compile(r"^\s+\*\s+SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
        scf_not_cvg_patt = re.compile(r"SCF NOT CONVERGED AFTER\s+\d+\s+CYCLES")

        for line in fout:
            if re.match(r"^\s+Total Charge\s+Charge\s+\.{4}", line):
                self.charge = int(line.split()[-1])

            elif energy_patt.match(line):
                energy = float(energy_patt.findall(line)[0])
                self.energies.append(energy)
                self.final_energy = energy

            elif m := scf_patt.match(line):
                self.scf_energies.append(float(m.group(1)))

            elif scf_cvg_patt.match(line):
                self.scf_converged = True

            elif scf_not_cvg_patt.search(line) or SCF_NOT_CONVERGED_MSG in line:
                self.scf_converged = False

            elif "SCF ITERATIONS" in line:
                # Start of an SCF block; mark unresolved until a converge/fail line.
                if self.scf_converged is None:
                    self.scf_converged = False

            elif dipole_patt.match(line):
                dipole = [float(val) for val in dipole_patt.findall(line)[0]]
                self.dipole_moment = dipole
                # The next two lines in ORCA output after "Total Dipole Moment" are usually blank or Magnitude
                try:
                    next(fout)
                    next(fout)
                    mag_line = next(fout)
                    if "Magnitude (Debye)" in mag_line:
                        self.dipole = float(mag_line.split()[-1])
                except (StopIteration, ValueError, IndexError):
                    pass

            elif "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                next(fout)  # Skip "-----------------------"
                line = next(fout, "")
                species = []
                coords = []
                while m := cart_patt.match(line):
                    specie = m.group(1)
                    if specie == "-":
                        specie = "X"
                    species.append(specie)
                    coords.append([float(val) for val in m.group(2, 3, 4)])
                    line = next(fout, "")

                if species:
                    mol = Molecule.from_sites(elements=species, sites=coords)
                    self.structures.append(mol)

            elif "THE OPTIMIZATION HAS CONVERGED" in line:
                self.optimization_converged = True

            elif "****ORCA TERMINATED NORMALLY****" in line:
                self.normal_termination = True

        # Content-level fallback for multiline / truncated LEANSCF messages.
        if SCF_NOT_CONVERGED_MSG in self.content:
            self.scf_converged = False
            self.normal_termination = False

        if len(self.structures) > 0:
            # MoleculeList.__getitem__ only serves the object cache for
            # non-negative indices; [-1] would return an ObjectId.
            self.final_structure = self.structures[len(self.structures) - 1]

    def get_qm_result(self) -> QMResult:
        """Deprecated sync helper — use :meth:`build_qm_result` instead."""
        raise RuntimeError(
            "OrcaOutput.get_qm_result() is async; use `await orca_output.build_qm_result()`"
        )

    async def build_qm_result(self, task_id: Optional[str] = None) -> QMResult:
        if self.qm_result is None:
            from molecular_qm_orca.lib.qm_result_from_orca import from_orca_output

            self.qm_result = await from_orca_output(self, task_id=task_id)
        return self.qm_result
