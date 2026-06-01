import re
import io
import logging
from pathlib import Path
from typing import List, Optional

from molecular_qm_models import QMResult
from molecular_qm_models.molecule import MoleculeList, Molecule
from simstack.core.node_runner import NodeRunner

logger = logging.getLogger(__name__)


class OrcaOutput:
    def __init__(self, node_runner: NodeRunner, filename: Path | str = "orca.out", ):
        self.node_runner = node_runner
        with open(filename, "r", encoding="utf-8", errors="replace") as f:
            self.content = f.read()

        self.charge: int = 0
        self.dipole: Optional[float] = None
        self.final_energy: Optional[float] = None
        self.dipole_moment: Optional[List[float]] = None
        self.energies: List[float] = []
        self.status: Optional[str] = None
        self.error: Optional[str] = None
        self.normal_termination: Optional[bool] = None

        self.scf_energies: List[float] = []
        self.scf_converged: Optional[bool] = None
        self.structures: MoleculeList = MoleculeList()
        self.final_structure: Optional[Molecule] = None

        self._parse()

        self.qm_result: QMResult | None = None

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

            elif "SCF ITERATIONS" in line:
                self.scf_converged = False

            elif dipole_patt.match(line):
                dipole = [float(val) for val in dipole_patt.findall(line)[0]]
                self.dipole_moment = dipole
                # The next two lines in ORCA output after "Total Dipole Moment" are usually blank or Magnitude
                # In deprecated/orca_output.py:
                # 1323:                fout.readline()
                # 1324:                fout.readline()
                # 1325:                self.dipole = float(fout.readline().split()[-1])
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

            elif "****ORCA TERMINATED NORMALLY****" in line:
                self.normal_termination = True

        if self.structures.molecules:
            self.final_structure = self.structures.molecules[-1]
        