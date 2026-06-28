# coding: utf-8

import pathlib
import re
import io
from enum import Enum, unique

import numpy as np
import pandas as pd
from pymatgen.electronic_structure.core import Spin
from pymatgen.core import Element
from pymatgen.core import Molecule as PymatgenMolecule
from molecular_qm_models import Molecule, BOHR_TO_ANGSTROM
import logging
logger = logging.getLogger(__name__)

"""
This module implements classes that represents input and output files
of the ORCA ab initio, DFT and semiempirical SCF-MO package.
"""

__author__ = "Hugo Santos-Silva, Germain Salvato Vallverdu"
__email__ = "germain.vallverdu@univ-pau.fr"
__copyright__ = "Copyright 2016, UPPA-CNRS"
__version__ = "2021.04.14"
__all__ = [
    "OrcaInput",
    "OrcaHessian",
    "OrcaVPT2",
    "OrcaEnGradfile",
    "OrcaOutfile",
    "OrcaOutput",
    "UA_TO_KCAL",
    "UA_TO_EV",
    "BOHR_TO_ANGS",
    "OrcaScanCalculation",
    "Scan_type",
]

# convertion Bohr --> Angstrom
BOHR_TO_ANGS = BOHR_TO_ANGSTROM  # Angstrom
UA_TO_KCAL = 627.5095  # kcal.mol-1
UA_TO_EV = 27.2114  # eV


class OrcaInput:
    """
    An object representing an Orca input file.
    """

    default_params = ["B3LYP Def2-SVP", "Opt"]

    def __init__(
            self,
            mol,
            charge=None,
            spin_multiplicity=None,
            input_parameters=None,
            blocks=None,
            with_geometry=False
    ):
        """
        Args:
            mol: Input molecule. If molecule is a single string, it is used
                as a direct input to the geometry section of the input file.
            charge (float): Charge of the molecule. If None, charge on
                molecule is used. Defaults to None. This allows the input file
                to be set a charge independently from the molecule itself.
            spin_multiplicity (float): Spin multiplicity of molecule. Defaults
                to None, which means that the spin multiplicity is set to 1
                if the molecule has no unpaired electrons and to 2 if there
                are unpaired electrons.
            input_parameters (list): Input parameters for run as a list
            blocks (list): Blocks input for advanced settings as a list of
                string. The block has to be already well formatted.
        """
        self._with_geometry = with_geometry
        if isinstance(mol, PymatgenMolecule):
            self._mol = mol
        elif isinstance(mol, Molecule):
            coords = []
            elements = []
            for atom in mol.atoms:
                coords.append(atom.position)
                elements.append(atom.element)
            self._mol = PymatgenMolecule(elements, coords)

        self.input_parameters = (
            input_parameters if input_parameters else self.default_params
        )
        self.blocks = blocks if blocks else []

        self.charge = charge if charge is not None else None
        self.spin_multiplicity = int(spin_multiplicity if spin_multiplicity is not None else None)
        if mol and isinstance(mol, PymatgenMolecule):
            if not self.charge and mol and hasattr(mol, "charge"):
                self.charge = mol.charge
            if not self.spin_multiplicity and mol and hasattr(mol, "spin_multiplicity"):
                self.spin_multiplicity = mol.spin_multiplicity

        if hasattr(self, "_mol"):
            self.molecule.set_charge_and_spin(self.charge, self.spin_multiplicity)


    @property
    def molecule(self):
        """
        Returns molecule associated with this OrcaInput.
        """
        return self._mol

    def get_string(self, with_geometry=True):
        """Return a string representation of the input file"""
        lines = ""
        # input parameters
        for param in self.input_parameters:
            lines += "! %s\n" % param
        # blocks
        for block in self.blocks:
            lines += block
        # geometry block
        if self._with_geometry:
            lines += "* xyz %d %d\n" % (self.charge, self.spin_multiplicity)
            if isinstance(self.molecule, str):
                lines += self.molecule
            elif isinstance(self.molecule, PymatgenMolecule):
                for site in self.molecule:
                    lines += "%2s" % site.specie
                    lines += "  %12.6f %12.6f %12.6f\n" % tuple(site.coords)
            elif isinstance(self.molecule, Molecule):
                for site in self.molecule:
                    lines += "%2s" % site.species
                    lines += "  %12.6f %12.6f %12.6f\n" % tuple(site.coords)
            else:
                raise TypeError(
                    "Bad Molecule object. I cannot export xyz coordinates."
                    "\nPlease, provides a string represenation or a pymatgen"
                    "Molecule object."
                )
            if lines[-1] != "\n":
                lines += "\n"
            lines += "*\n"
        return lines

    def __str__(self):
        return self.get_string()


class OrcaHessian:
    """
    Parser for ORCA Hessian file. All data are in atomic units.

    .. attribute:: filename

        Path to the ORCA Hessian file

    .. attribute:: energy

        Energy in Hartree.

    .. attribute:: molecule

        The molecule geometry read in the hessian file. Coordinates are read
        in atomic unit and store in the molecule object in atomic units.

    .. attribute:: frequencies

        A list of dict for each frequencies with ::

            {
                "frequency": freq in cm-1,
                "symmetry": symmetry tag
                "r_mass": Reduce mass,
                "f_constant": force constant,
                "IR_intensity": IR Intensity,
                "mode": normal mode
             }

        The normal mode is a 1D vector of dx, dy dz of each atom.

    .. attribute:: hessian

        Matrix of second derivatives of the energy with respect to cartesian
        coordinates in the **input orientation** frame. Need #P in the
        route section in order to be in the output.

    """

    def __init__(self, filename):
        """The class reads an orca hessian file.

        Args:
            filename: Filename of ORCA hessian file.
        """
        self.filename = filename

        # set all attributes to None, thus it will always exist
        self.molecule = None
        self.frequencies = None
        self.hessian = None
        self.energy = None

        # parse file
        self._parse()

    def _parse(self):
        """Parse the file and fill in object and attributes"""

        frequencies = []
        normal_modes = None

        with open(self.filename, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                #
                # Read energy
                #
                if "$act_energy" in line:
                    self.energy = float(f.readline())

                #
                # Read the hessian matrix
                #
                elif "$hessian" in line:
                    dimension = int(f.readline())
                    matrix = np.zeros((dimension, dimension))
                    while line != "\n":
                        line = f.readline()
                        if "." not in line:
                            jindex = [int(jj) for jj in line.split()]
                        else:
                            iindex = int(line.split()[0])
                            datas = [float(val) for val in line.split()[1:]]
                            for j, data in enumerate(datas):
                                # print (j, jindex[j], iindex)
                                matrix[iindex, jindex[j]] = data

                    self.hessian = matrix

                #
                # parse normal mode
                #
                elif "$normal_modes" in line:
                    dimension = int(f.readline().split()[0])
                    normal_modes = np.zeros((dimension, dimension))
                    end = False
                    while not end:
                        jindex = [int(j) for j in f.readline().split()]
                        for i in range(dimension):
                            val = [float(v) for v in f.readline().split()[1:]]
                            for jval, j in enumerate(jindex):
                                normal_modes[i, j] = val[jval]
                        if jindex[-1] == dimension - 1:
                            end = True

                #
                # Parse geometry
                #
                elif "$atoms" in line:
                    atoms = []
                    coords = []

                    natoms = int(next(f))
                    for i in range(natoms):
                        line = next(f)
                        data = line.split()
                        atoms.append(Element(data[0]))
                        coords.append([float(x) for x in data[2:5]])
                    self.molecule = PymatgenMolecule(atoms, coords)

                #
                # Parse vibrational frequencies
                #
                elif "$vibrational_frequencies" in line:
                    ntfreqs = int(f.readline())
                    for i in range(ntfreqs):
                        data = f.readline().split()
                        frequencies.append(
                            {
                                "frequency": float(data[1]),
                                "r_mass": None,
                                "f_constant": None,
                                "IR_intensity": None,
                                "symmetry": None,
                                "mode": [],
                            }
                        )

                #
                # Parse IR data
                #
                elif "$ir_spectrum" in line:
                    ntfreqs = int(f.readline())  # aussi int(next(f))
                    for i in range(ntfreqs):
                        data = f.readline().split()
                        if len(frequencies) > i:
                            frequencies[i]["IR_intensity"] = float(data[1])
                        else:
                            frequencies.append(
                                {
                                    "frequency": float(data[0]),
                                    "r_mass": None,
                                    "f_constant": None,
                                    "IR_intensity": float(data[1]),
                                    "symmetry": None,
                                    "mode": [],
                                }
                            )

        # add normal modes to frequencies dict
        if normal_modes is not None:
            for ifreq, freq in enumerate(frequencies):
                freq["mode"] = normal_modes[:, ifreq]
        self.frequencies = frequencies

    def symmetrize_hessian(self, inplace=False):
        """
        Return a symmetrized hessian matrix.

        Args:
            inplace (bool): if True, do operation inplace and return None.
        """
        matrix = (self.hessian + self.hessian.T) / 2
        if inplace:
            self.hessian = matrix
            return None
        else:
            return matrix

    def get_mol_ang(self, inplace=False):
        """
        Return a new PymatgenMolecule object with coordinate in angstrom

        Args:
            inplace (bool): if True, do operation inplace and return None.
        """
        newmol = PymatgenMolecule(
            self.molecule.species, self.molecule.cart_coords * BOHR_TO_ANGS
        )
        if inplace:
            self.molecule = newmol
            return None
        else:
            return newmol


class OrcaVPT2:
    """
    Parser for ORCA .vpt2 file.

    .. attribute:: filename

        Path to the ORCA .vpt2 file

    .. attribute:: settings

        VPT2 calculation settings.

    .. attribute:: molecule

        The molecule geometry read in the hessian file. Coordinates are read
        in atomic unit and store in the molecule object in atomic units.

    .. attribute:: hessian

        Hessian matrix in Eh/(bohr**2). Array shape (3N, 3N)

    .. attribute:: dipole_derivative

        Dipole derivatives in (Eh * bohr)^(1/2). Array shape (3N, 3)

    .. attribute:: cubic_terms

        Cubic force field in cm-1. Array shape (3N-6, 3N-6, 3N-6)

    .. attribute:: semi_quartic_terms

        Semi-quartic force field in cm-1. Only [i][j][k][k] terms are provided.
        The last index is not repeated. Thus the shape of the array is
        (3N-6, 3N-6, 3N-6)

    """

    def __init__(self, filename):
        """The class reads an ORCA vpt2 file.

        Args:
            filename: Filename of ORCA .vpt2 file.
        """
        self.filename = filename

        # set all attributes to None, thus it will always exist
        self.settings = None
        self.molecule = None
        self.hessian = None
        self.dipole_derivative = None
        self.cubic_terms = None
        self.semi_quartic_terms = None

        # parse file
        self._parse()

    def _parse(self):
        """Parse the file and fill in object and attributes"""

        num_patt = re.compile(r"\d\.\d+[deDE]?[+-]?\d+")
        hess_patt = re.compile(r"^\s+(\d+)\s+(\d+)\s+([+-]?\d+\.\d+)")
        cubic_patt = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([+-]?\d+\.\d+)")

        with open(self.filename, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                #
                # Read energy
                #
                if "# VPT2 settings" in line:
                    self.settings = dict()
                    line = f.readline()
                    while line.strip() != "":
                        key, val = line.split("=")
                        if num_patt.match(val):
                            val = float(val)
                        elif re.match(r"\d+", val):
                            val = int(val)
                        self.settings[key] = val
                        line = f.readline()

                if "# Atomic coordinates in Angstroem" in line:
                    line = f.readline()
                    species = list()
                    coords = list()
                    line = f.readline()
                    while re.match(
                            r"^\s+\w{1,2}\s+\d+\s+\d+\.\d+\s+[+-]?\d+\.\d+", line
                    ):
                        data = line.split()
                        species.append(data[0])
                        coords.append([float(val) for val in data[3:6]])
                        line = f.readline()
                    self.molecule = PymatgenMolecule(species, coords)

                if "# Hessian[i][j] in Eh/(bohr**2)" in line:
                    ni, nj = [int(val) for val in f.readline().split()]
                    self.hessian = np.zeros((ni, nj))
                    line = f.readline()
                    while hess_patt.match(line):
                        i, j, val = hess_patt.match(line).groups()
                        self.hessian[int(i), int(j)] = float(val)
                        line = f.readline()

                if "# Dipole Derivatives[i][j] in (Eh*bohr)^1/2" in line:
                    ni, nj = [int(val) for val in f.readline().split()]
                    self.dipole_derivative = np.zeros((ni, nj))
                    line = f.readline()
                    while hess_patt.match(line):
                        i, j, val = hess_patt.match(line).groups()
                        self.dipole_derivative[int(i), int(j)] = float(val)
                        line = f.readline()

                if "# Cubic[i][j][k] force field in 1/cm" in line:
                    ni, nj, nk = [int(val) for val in f.readline().split()]
                    self.cubic_terms = np.zeros((ni, nj, nk))
                    line = f.readline()
                    while cubic_patt.match(line):
                        i, j, k, val = cubic_patt.match(line).groups()
                        self.cubic_terms[int(i), int(j), int(k)] = float(val)
                        line = f.readline()

                if "# Semi-quartic[i][j][k][k] force field in 1/cm" in line:
                    ni, nj, nk = [int(val) for val in f.readline().split()]
                    self.semi_quartic_terms = np.zeros((ni, nj, nk))
                    line = f.readline()
                    while cubic_patt.match(line):
                        i, j, k, val = cubic_patt.match(line).groups()
                        self.semi_quartic_terms[int(i), int(j), int(k)] = float(val)
                        line = f.readline()

    def symmetrize_hessian(self, inplace=False):
        """
        Return a symmetrized hessian matrix.

        Args:
            inplace (bool): if True, do operation inplace and return None.
        """
        matrix = (self.hessian + self.hessian.T) / 2
        if inplace:
            self.hessian = matrix
            return None
        else:
            return matrix

    def get_mol_ang(self, inplace=False):
        """
        Return a new Molecule object with coordinate in angstrom

        Args:
            inplace (bool): if True, do operation inplace and return None.
        """
        newmol = PymatgenMolecule(
            self.molecule.species, self.molecule.cart_coords * BOHR_TO_ANGS
        )
        if inplace:
            self.molecule = newmol
            return None
        else:
            return newmol


class OrcaOutfile:
    """
    Parser for ORCA output file. (WORKS FOR WHICH VERSIONS???)

    .. attribute:: number_electrons

        Total number of electrons in the system

    .. attribute:: number_basis_functions

        Number of contracted basis functions in the main gaussian basis set.

    .. attribute:: nuclear_repulsion

        Nuclear repulsion energy of the last geometry.

    .. attribute:: energies

        List of all FINAL SINGLE POINT ENERGY values. Contrary to the
        `scf_energies`, these values include the various corrections that
        may have been requiered in the calculations(dispersion for example).

    .. attribute:: scf_energies

        List of all Total Energy values, obtained at the end of the SCF
        cylce.

    .. attribute:: charge

        Total charge of the molecule.

    .. attribute:: spin_multiplicity

        Spin multiplicity of the molecule.

    .. attribute:: mul_charges

        List of the last Mulliken atomic charges. If a geometry optimization
        is done, only the last charges are saved.

    .. attribute:: loe_charges

        List of the Loewdin atomic charges. If a geometry optimization is done,
        only the last charges are saved.

    .. attribute:: hirshfeld_charges

        List of the Hirshfeld atomic charges. If a geometry optimization is done,
        only the last charges are saved.

    .. attribute:: esp_charges

        List of the ESP charges fitted on the electrostatic potential. Use CHELPG
        in the input file to get that charges.

    .. attribute:: mul_spin_pop

        List of the last Mulliken spin populations. If a geometry optimization
        is done, only the last spin populations are saved.

    .. attribute:: loe_spin_pop

        List of the last Loewdin spin populations. If a geometry optimization
        is done, only the last spin populations are saved.

    .. attribute:: hirshfeld_spin_pop

        List of the last Hirshfeld spin populations. If a geometry optimization
        is done, only the last spin populations are saved.

    .. attribute:: orca_input

        An OrcaInput instance created from the input file read on the Orca
        outfile. The considered structure is the last one.

    .. attribute:: bond_orders

        Dict of bond order values read in the output file such as:
        {(0, 1): 0.8709, (1, 6): 1.234, ...}

    .. attribute:: dipole_moment

        Components of the dipole moment in a.u.

    .. attribute:: dipole

        Magnitude of the dipole moment in Debye

    .. attribute:: structures

        List of all structures in the calculation as pymatgen.Molecule object.
        The last geometry for the last additionnal SCF calculation is included.

    .. attribute:: is_spin

        True if the calculation is a unrestricted calculation (UKS) in INPUT. False
        if the calculation is a restricted calculation.

    .. attribute:: optimization_converged

        None if Opt is not in INPUT. True if optimization converge else False.

    .. attribute:: normal_termination

        True if Orca terminated normally.

    .. attribute: thermochemistry

        bool, if True, calculation of thermochemistry quantity was done and read.

    .. attribute:: temperature

        Temperature, in Kelvin, for the calculations of thermochemical quantities

    .. attribute:: pressure

        Pressure, in atmosphere, for the calculations of thermochemical quantities

    .. attribute:: mass

        Total mass, in AMU (g.mol-1), for the calculations of themochemical quantities

    .. attribute:: frequencies

        List of vibrational frequencies in cm-1

    .. attribute:: thermo_data

        dictionnary of all thermochemistry data read in output file both in ua and
        kcal.mol-1. The first value is in ua and the second one in kcal.mol-1.

        ['electronic energy', 'zero point energy', 'thermal vibrational correction',
         'thermal rotational correction', 'thermal translational correction',
         'inner energy', 'thermal enthalpy correction', 'total enthalpy',
         'electronic entropy', 'vibrational entropy', 'rotational entropy',
         'translational entropy', 'total entropy correction',
         'final gibbs free enthalpy', 'g-e(el)', 'entropy']

    .. attribute:: mo_energies

        Dict of the molecular orbital energies (in Hartree) for each
        spin (if relevant) such as:

            {Spin.up: energies, Spin.down: energies}

    .. attribute:: mo_occupations

        Dict of molecular orbital occupations for each spin (if relevant)
        such as:

            {Spin.up: occupations, Spin.down: occupations}

    .. attribute:: mo_matrix

        Dict of arrays of molecular orbital coefficients for each spin
        (if relevant). The shapes of the arrays are
        `(number_basis_functions x number_basis_functions)`. The dict is
        such as:

            {Spin.up: coefficients, Spin.down: coefficients}

    .. attribute: overlap_matrix

        Overlap matrix of the basis orbitals. The shape of the matrix
        is `(number_basis_functions x number_basis_functions)`.

    .. attribute quadrupole_moment

        TODO: looks like the following


    """

    def __init__(self, filename, node_runner=None):
        """The class reads the ORCA output file and store the data in
        the attributes' instance.

        Args:
            filename: Filename or file object of a Orca output file.
            node_runner (optional): A node runner instance for logging.
        """
        self.node_runner = node_runner
        # initialisation of all attributes, thus they will always exist
        self.mul_charges = None
        self.mul_spin_pop = None
        self.loe_charges = None
        self.loe_spin_pop = None
        self.esp_charges = None
        self.hirshfeld_charges = None
        self.hirshfeld_spin_pop = None
        self.bond_orders = None
        self.dipole_moment = None
        self.dipole = None
        self.energies = list()
        self.scf_energies = list()
        self.structures = list()
        self.charge = None
        self.spin_multiplicity = None
        self.is_spin = False
        self.optimization_converged = None
        self.normal_termination = False
        self.scf_converge = False
        self.number_electrons = None
        self.number_basis_functions = None
        self.nuclear_repulsion = None

        self.thermochemistry = False
        self.temperature = None
        self.pressure = None
        self.mass = None
        self.frequencies = None
        self.thermo_data = None

        self.mo_energies = None
        self._at_index = None
        self._ao_name = None
        self._elements = None
        self.mo_occupations = None
        self.mo_matrix = None
        self._homo = None
        self._lumo = None

        self.overlap_matrix = None

        # Electronic properties from %elprop section / property blocks
        # These are parsed on demand from the output if present.
        self._quadrupole_moment = None
        self._polarizability_dipole = None
        self._polarizability_dipole_quadrupole = None
        self._polarizability_dipole_quadrupole_traceless = None
        self._polarizability_velocity = None
        self._hyperpolarizability = None

        # Settings parsed from the %elprop block in the input section
        # e.g. {"Dipole": True, "Quadrupole": True, ...}
        self._elprop_settings = {}

        # parse file
        try:
            # try to open file
            with open(filename, "r", encoding="utf-8", errors="replace") as fout:
                self._parse(fout)
        except TypeError:
            # parse the file object
            self._parse(filename)

    def error(self, msg):
        """Log an error message"""
        if self.node_runner:
            self.node_runner.error(msg)
        else:
            logger.error(msg)

    def info(self, msg):
        """Log an info message"""
        if self.node_runner:
            self.node_runner.info(msg)
        else:
            logger.info(msg)

    def debug(self, msg):
        """Log a debug message"""
        if self.node_runner:
            self.node_runner.debug(msg)
        else:
            logger.debug(msg)

    def warning(self, msg):
        """Log a warning message"""
        if self.node_runner:
            self.node_runner.warning(msg)
        else:
            logger.warning(msg)

    @property
    def final_energy(self):
        """Last SCF energy"""
        if self.scf_energies and len(self.scf_energies) > 0:
            return self.energies[-1]
        else:
            return None

    @property
    def initial_structure(self):
        """first geometry"""
        return self.structures[0]

    @property
    def final_structure(self):
        """last geometry"""
        if self.structures and len(self.structures) > 0:
            return self.structures[-1]
        else:
            return None

    @property
    def gibbs_free_enthalpy(self):
        """Gibbs free enthalpy in kcal.mol-1"""
        if self.thermochemistry:
            return self.thermo_data["final gibbs free enthalpy"][1]
        else:
            self.warning("no thermochemistry data available.")
            return None

    @property
    def homo(self):
        """Number and energy, in Hartree, of the highest occupied
        molecular orbital (HOMO). The number of the first molecular
        orbital is zero."""
        return self._homo

    @property
    def lumo(self):
        """Number and energy, in Hartree, of the lowest unoccupied
        molecular orbital (LUMO). The number of the first molecular
        orbital is zero."""
        return self._lumo

    @property
    def mo_dataframe(self):
        """A dict of data frames with MO coefficients such as:

            {Spin.up: dataframe, Spin.down: dataframe}

        Each column of the data frame is a MO. Each row contains the
        coefficients for an atom, in a specfic AO. The numbers of the
        MO start at 0.
        """
        if self.mo_matrix is None:
            # return an empty dataFrame
            return {Spin.up: pd.DataFrame()}

        ao_type = [ao[1] for ao in self._ao_name]
        index = pd.MultiIndex.from_arrays(
            [self._at_index, self._elements, self._ao_name, ao_type],
            names=["iat", "Element", "AO", "shell"],
        )
        columns = [f"MO_{i}" for i in range(self.number_basis_functions)]
        if self.is_spin:
            return {
                Spin.up: pd.DataFrame(
                    self.mo_matrix[Spin.up], index=index, columns=columns
                ),
                Spin.down: pd.DataFrame(
                    self.mo_matrix[Spin.down], index=index, columns=columns
                ),
            }
        else:
            return {
                Spin.up: pd.DataFrame(
                    self.mo_matrix[Spin.up], index=index, columns=columns
                )
            }

    def error(self, msg):
        """Log an error message"""
        if self.node_runner:
            self.node_runner.error(msg)
        else:
            logger.error(msg)

    def info(self, msg):
        """Log an info message"""
        if self.node_runner:
            self.node_runner.info(msg)
        else:
            logger.info(msg)

    def debug(self, msg):
        """Log a debug message"""
        if self.node_runner:
            self.node_runner.debug(msg)
        else:
            logger.debug(msg)

    def warning(self, msg):
        """Log a warning message"""
        if self.node_runner:
            self.node_runner.warning(msg)
        else:
            logger.warning(msg)

    # ---------------------------------------------------------------------
    # Helper methods for electronic property handling
    # ---------------------------------------------------------------------

    def _parse_elprop_settings(self):
        """Parse the %elprop block from the reconstructed input.

        Populates ``self._elprop_settings`` with entries like
        ``{"Dipole": True, "Quadrupole": False, ...}``.
        """
        self._elprop_settings = {}

        if not hasattr(self, "input") or not self.input:
            return

        in_block = False
        # matches e.g. "Dipole true", optionally followed by comments
        line_patt = re.compile(r"^(?P<name>\w+)\s+(?P<val>true|false)\b",
                               re.IGNORECASE)

        for raw_line in self.input.splitlines():
            line = raw_line.strip()
            lower = line.lower()
            if lower.startswith("%elprop"):
                in_block = True
                continue
            if not in_block:
                continue
            if not line:
                # skip empty lines inside block
                continue
            if lower.startswith("end"):
                # end of %elprop block
                break
            # strip inline comments starting with '#'
            if "#" in line:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue

            m = line_patt.match(line)
            if not m:
                continue
            name = m.group("name")
            val = m.group("val").lower() == "true"
            self._elprop_settings[name] = val

    def _is_elprop_enabled(self, key: str) -> bool:
        """Return True if a given elprop keyword was enabled in the input.

        Parameters
        ----------
        key
            Name of the elprop keyword, e.g. ``"Dipole"``, ``"Quadrupole"``.
        """
        return bool(self._elprop_settings.get(key, False))

    def _parse(self, fout):
        """Parse the file and fill in object and attributes"""

        kw_patt = re.compile(r"^\|\s+\d+>\s!\s(.+)")
        input_patt = re.compile(r"^\|\s+\d+>\s+(.+)")
        q_patt = re.compile(
            r"^\s*\d+\s*([a-zA-Z]+)\s*:\s+([+-]?\d+\.\d+)\s*([+-]?\d+\.\d+)?"
        )
        float_patt = re.compile(r"\s*([+-]?\d+\.\d+)")
        bo_patt = re.compile(
            r"B\(\s*(\d+)-(\S{1,2})\s*,\s*(\d+)-(\S{1,2})\s*\)\s+:\s*(\d+\.\d+)"
        )
        dipole_patt = re.compile(
            r"^Total Dipole Moment\s+:\s+([+-]?\d+\.\d+)"
            r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
        )
        hirsh_patt = re.compile(
            r"\s+\d+\s+[a-zA-Z]{1,2}\s+([+-]?\d+\.\d+)" r"\s+[+-]?\d+\.\d+"
        )
        energy_patt = re.compile(r"FINAL SINGLE POINT ENERGY\s+([+-]?\d+\.\d+)")
        xtb_charge_patt = re.compile(r":\s+net charge\s+([+-]?\d+)\s+:")
        xtb_mult_patt = re.compile(r":\s+spin multiplicity\s+(\d+)\s+:")
        xtb_unpaired_patt = re.compile(r":\s+unpaired electrons\s+(\d+)\s+:")
        scf_patt = re.compile(
            r"Total Energy\s+:\s+([+|-]?\d+\.\d+) Eh\s+([+|-]?\d+\.\d+) eV"
        )
        xtb_scf_patt = re.compile(
            r"^\s+\d+\s+([+-]?\d+\.\d+)\s+[+-]?\d+\.\d+E[+-]\d+.*"
        )
        cart_patt = re.compile(
            r"^\s+([a-zA-Z]{1,2}|-)\s+([+-]?\d+\.\d+)"
            r"\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)"
        )
        scf_cvg_patt = re.compile(r"^\s+\*\s+SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")
        xtb_scf_cvg_patt = re.compile(r"\*\*\* convergence criteria satisfied after \d+ iterations \*\*\*")
        mo_patt = re.compile(
            r"^\s*(?P<iat>\d+)(?P<element>[a-zA-Z]+)"
            r"\s+(?P<AO>[\w+-]{1,6})"
            r"(?P<coeffs>(\s*[+-]?\d+\.\d+){1,6})"
        )
        thermo_patt = re.compile(
            r"^(?P<_name>.+)\.\.\.\s+(?P<uaval>.[+-]?\d+\.\d+)\sEh"
            r"(\s+(?P<kcalval>.[+-]?\d+\.\d+)\skcal/mol)?"
        )
        freq_patt = re.compile(r"freq\.\s+([+-]?\d+\.\d+)\s+E\(vib\)\s+")

        calculation_start = False
        # look for calculation starts
        for line in fout:
            if "INPUT FILE" in line:
                #
                # Read input file
                #

                # set up list to store input parameters
                input_parameters = list()
                blocks = ""
                self.input = ""

                line = fout.readline()
                while "**END OF INPUT**" not in line:
                    if "#" in line:
                        line = fout.readline()
                        continue
                    elif kw_patt.search(line):
                        data = kw_patt.findall(line)[0]
                        self.input += "! " + data + "\n"
                        input_parameters += [k.lower() for k in data.split()]
                    elif m := re.match(r"^\|\s+\d+>\s*(\*.*)", line):
                        self.input += m.group(1)
                        self.input += "\n"
                    elif m := input_patt.match(line):
                        data = m.group(1) + "\n"
                        self.input += data
                        blocks += data
                    line = fout.readline()

                if "opt" in input_parameters:
                    self.optimization_converged = False

                # Extract %elprop settings from the reconstructed input
                # so that property accessors know which quantities are
                # expected to be present.
                self._parse_elprop_settings()

                calculation_start = True
                break

        if not calculation_start:
            return None

        # read data in the output file
        for line in fout:
            if re.match(r"^\s+Hartree-Fock type\s+HFTyp\s+\.{4}", line):
                if line.split()[-1] == "UHF":
                    self.is_spin = True

            elif re.match(r"^\s+Number of Electrons\s+NEL\s+\.{4}", line):
                self.number_electrons = int(line.split()[-1])

            elif re.match(r"^\s+Basis Dimension\s+Dim\s+\.{4}", line):
                self.number_basis_functions = int(line.split()[-1])

            elif re.match(r"^\s+Total Charge\s+Charge\s+\.{4}", line):
                self.charge = int(line.split()[-1])

            elif re.match(r"^\s+Multiplicity\s+Mult\s+\.{4}", line):
                self.spin_multiplicity = int(line.split()[-1])

            elif re.match(r"^\s+Nuclear Repulsion\s+ENuc\s+\.{4}", line):
                self.nuclear_repulsion = float(line.split()[-2])

            elif m := xtb_charge_patt.search(line):
                self.charge = int(m.group(1))

            elif m := xtb_mult_patt.search(line):
                self.spin_multiplicity = int(m.group(1))

            elif m := xtb_unpaired_patt.search(line):
                self.spin_multiplicity = 2 * int(m.group(1)) + 1

            elif energy_patt.match(line):
                energy = float(energy_patt.findall(line)[0])
                self.energies.append(energy)



            elif m := scf_patt.match(line):
                self.scf_energies.append(float(m.group(1)))

            elif m := xtb_scf_patt.match(line):
                self.scf_energies.append(float(m.group(1)))

            elif scf_cvg_patt.match(line):
                self.scf_converge = True

            elif xtb_scf_cvg_patt.search(line):
                self.scf_converge = True

            elif "SCF ITERATIONS" in line:
                # start new SCF step
                self.scf_converge = False

            elif "OVERLAP MATRIX" == line.strip():
                fout.readline()
                self.overlap_matrix = np.zeros(
                    (self.number_basis_functions, self.number_basis_functions), np.float
                )

                n = 0
                while n < self.number_basis_functions - 1:
                    line = fout.readline()  # basis function index
                    jbf = np.array(line.split(), np.int64)

                    ibf = 0
                    while ibf < self.number_basis_functions - 1:
                        line = fout.readline().split()
                        ibf = int(line[0])
                        Sij = np.array(line[1:], np.float)
                        self.overlap_matrix[ibf, jbf] = Sij

                    n = jbf[-1]

            elif "ORBITAL ENERGIES" in line:
                fout.readline()
                fout.readline()
                fout.readline()

                energies = list()
                occupations = list()

                while len(energies) < self.number_basis_functions:
                    line = fout.readline()
                    if not line:
                        break
                    if line.startswith("NO") or "E(Eh)" in line:
                        continue
                    if line.startswith("*Only"):
                        break
                    if "---" in line:
                        continue
                    if ":" in line:
                        continue
                    if "MULLIKEN" in line or "SPIN DOWN" in line:
                        break

                    parts = line.split()
                    if not parts:
                        continue

                    try:
                        # Standard ORCA: # Occupation Energy/Eh Energy/eV
                        # 0: #, 1: Occupation, 2: Energy/Eh, 3: Energy/eV

                        # xtb pattern can be:
                        # 7        2.0000           -0.4352178             -11.8429 (HOMO)
                        # 8                          0.1235668               3.3624 (LUMO)

                        # We can distinguish by looking at the positions of numbers.
                        # Occupation is usually around column 10-20.
                        # Energy/Eh is around column 30-40.

                        # Let's check the raw line to see if there is a gap where occupation should be.
                        # The occupation column usually starts around index 10 and ends around 20-25.

                        occupation_slice = line[10:26].strip()
                        if occupation_slice == "":
                            # Likely xtb pattern without occupation
                            occupations.append(0.0)
                            energies.append(float(parts[1]))
                        else:
                            # Likely standard pattern or xtb with occupation
                            occupations.append(float(parts[1]))
                            energies.append(float(parts[2]))
                    except (ValueError, IndexError):
                        if not any(c.isdigit() for c in parts[0]):
                             # skip lines that don't start with a number (likely labels)
                             continue
                        self.error("Could not parse: " + line)

                self.mo_energies = {Spin.up: np.array(energies)}
                self.mo_occupations = {Spin.up: np.array(occupations)}
                idx_homo = np.where(np.array(occupations) > 0.0)[0][-1]
                idx_lumo = idx_homo + 1

                e_homo = -1
                e_lumo = -1

                if self.is_spin:
                    fout.readline()
                    fout.readline()
                    fout.readline()

                    energies = list()
                    occupations = list()
                    while len(energies) < self.number_basis_functions:
                        line = fout.readline()
                        if not line:
                            break
                        if line.startswith("NO") or "E(Eh)" in line:
                            continue
                        if "---" in line:
                            continue
                        if ":" in line:
                            continue
                        if "MULLIKEN" in line:
                            break
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            energies.append(float(parts[2]))
                            occupations.append(float(parts[1]))
                        except (ValueError, IndexError):
                            if not any(c.isdigit() for c in parts[0]):
                                # skip lines that don't start with a number (likely labels)
                                continue
                            self.error("Could not parse: " + line)
                    self.mo_energies[Spin.down] = np.array(energies)
                    self.mo_occupations[Spin.down] = np.array(occupations)

                    try:
                        idx_down = np.where(np.array(occupations) > 0.0)[0][-1]
                        if idx_down > idx_homo:
                            idx_homo = idx_down
                            e_homo = self.mo_energies[Spin.down][idx_homo]
                        if idx_down + 1 < idx_lumo:
                            idx_lumo = idx_down
                            e_lumo = self.mo_energies[Spin.down][idx_lumo]
                    except Exception as e:
                        logger.error("Could not parse: " + str(occupations))
                        logger.error("Error details: " + str(e))
                else:
                    e_homo = self.mo_energies[Spin.up][idx_homo]
                    e_lumo = self.mo_energies[Spin.up][idx_lumo]

                self._homo = (idx_homo, e_homo)
                self._lumo = (idx_lumo, e_lumo)


            elif "MOLECULAR ORBITALS" == line.strip():
                spins = [Spin.up]
                if self.is_spin:
                    spins.append(Spin.down)

                fout.readline()

                self.mo_matrix = {}
                for spin in spins:
                    self.mo_matrix[spin] = np.zeros(
                        (self.number_basis_functions, self.number_basis_functions)
                    )

                    n_mo = 0
                    while n_mo < self.number_basis_functions - 1:
                        line = fout.readline()  # MO index
                        jbf = np.array(line.split(), np.int64)

                        fout.readline()  # MO energies
                        fout.readline()  # MO occupation
                        fout.readline()  # dash line

                        self._at_index = list()
                        self._ao_name = list()
                        self._elements = list()

                        ibf = 0
                        while ibf < self.number_basis_functions:
                            data = mo_patt.match(fout.readline()).groupdict()
                            coeffs = float_patt.findall(data.pop("coeffs"))
                            self.mo_matrix[spin][ibf, jbf] = [
                                float(coeff) for coeff in coeffs
                            ]
                            self._elements.append(data["element"])
                            self._at_index.append(int(data["iat"]))
                            self._ao_name.append(data["AO"])
                            ibf += 1

                        n_mo = jbf[-1]

                    fout.readline()  # read blank line between spins

            elif "MULLIKEN ATOMIC CHARGES" in line:
                fout.readline()
                line = fout.readline()
                self.mul_charges = list()
                if self.is_spin:
                    self.mul_spin_pop = list()
                while m := q_patt.match(line):
                    self.mul_charges.append(float(m.group(2)))
                    if self.is_spin:
                        self.mul_spin_pop.append(float(m.group(3)))
                    line = fout.readline()

            elif "LOEWDIN ATOMIC CHARGES" in line:
                fout.readline()
                line = fout.readline()
                self.loe_charges = list()
                if self.is_spin:
                    self.loe_spin_pop = list()
                while m := q_patt.search(line):
                    self.loe_charges.append(float(m.group(2)))
                    if self.is_spin:
                        self.loe_spin_pop.append(float(m.group(3)))
                    line = fout.readline()

            elif "CHELPG Charges" in line:
                fout.readline()
                line = fout.readline()
                self.esp_charges = []
                while q_patt.search(line):
                    vals = [float(c) for c in float_patt.findall(line)]
                    self.esp_charges.append(vals[0])
                    line = fout.readline()

            elif "HIRSHFELD ANALYSIS" in line:
                [fout.readline() for _ in range(6)]
                line = fout.readline()
                self.hirshfeld_charges = list()
                while hirsh_patt.match(line):
                    val = float(hirsh_patt.findall(line)[0])
                    self.hirshfeld_charges.append(val)
                    line = fout.readline()

            elif "Mayer bond orders larger than 0.1" in line:
                self.bond_orders = {}
                line = fout.readline()
                while bo_patt.match(line):
                    bo_line = bo_patt.findall(line)
                    for iat, _, jat, _, val in bo_line:
                        key = (int(iat), int(jat))
                        self.bond_orders[key] = float(val)
                    line = fout.readline()

            elif dipole_patt.match(line):
                dipole = [float(val) for val in dipole_patt.findall(line)[0]]
                self.dipole_moment = np.array(dipole)
                fout.readline()
                fout.readline()
                self.dipole = float(fout.readline().split()[-1])

            # --------------------
            # QUADRUPOLE MOMENT
            # --------------------
            elif line.strip() == "QUADRUPOLE MOMENT":
                try:
                    # Skip header and move to the column labels line (XX YY ZZ ...)
                    while True:
                        line = fout.readline()
                        if not line:
                            break
                        if line.strip().startswith("XX"):
                            break

                    # Read until we find the TOT (a.u.) line and the
                    # corresponding Buckingham line just below it.
                    tot_au = tot_buck = None
                    while True:
                        line = fout.readline()
                        if not line:
                            break
                        if line.lstrip().startswith("TOT"):
                            vals_au = [float(v) for v in float_patt.findall(line)[:6]]
                            buck_line = fout.readline()
                            vals_buck = [float(v) for v in float_patt.findall(buck_line)[:6]]
                            if len(vals_au) == 6 and len(vals_buck) == 6:
                                tot_au = vals_au
                                tot_buck = vals_buck
                            break

                    # Build 3x3 tensors from XX, YY, ZZ, XY, XZ, YZ
                    if tot_au is not None and tot_buck is not None:
                        xx, yy, zz, xy, xz, yz = tot_au
                        raw_au = np.array(
                            [
                                [xx, xy, xz],
                                [xy, yy, yz],
                                [xz, yz, zz],
                            ]
                        )
                        xx_b, yy_b, zz_b, xy_b, xz_b, yz_b = tot_buck
                        raw_buck = np.array(
                            [
                                [xx_b, xy_b, xz_b],
                                [xy_b, yy_b, yz_b],
                                [xz_b, yz_b, zz_b],
                            ]
                        )
                    else:
                        raw_au = raw_buck = None

                    # Find "diagonalized tensor:" block
                    while line and "diagonalized tensor" not in line.lower():
                        line = fout.readline()
                        if not line:
                            break

                    diag_au = diag_buck = traceless_buck = None
                    if line and "diagonalized tensor" in line.lower():
                        line = fout.readline()
                        diag_au_vals = [float(v) for v in float_patt.findall(line)[:3]]
                        line = fout.readline()
                        diag_buck_vals = [float(v) for v in float_patt.findall(line)[:3]]
                        line = fout.readline()
                        traceless_vals = [float(v) for v in float_patt.findall(line)[:3]]
                        if len(diag_au_vals) == 3:
                            diag_au = np.array(diag_au_vals)
                        if len(diag_buck_vals) == 3:
                            diag_buck = np.array(diag_buck_vals)
                        if len(traceless_vals) == 3:
                            traceless_buck = np.array(traceless_vals)

                    # Orientation matrix (3x3)
                    orientation = None
                    # skip blank line(s)
                    line = fout.readline()
                    while line and not line.strip():
                        line = fout.readline()
                    if line:
                        rows = []
                        # first row already read into `line`
                        rows.append([float(v) for v in float_patt.findall(line)[:3]])
                        for _ in range(2):
                            line = fout.readline()
                            if not line:
                                break
                            rows.append([float(v) for v in float_patt.findall(line)[:3]])
                        if len(rows) == 3 and all(len(r) == 3 for r in rows):
                            orientation = np.array(rows)

                    # Find isotropic quadrupole
                    isotropic = None
                    while line and "Isotropic quadrupole" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    if line and "Isotropic quadrupole" in line:
                        vals = float_patt.findall(line)
                        if vals:
                            isotropic = float(vals[-1])

                    self._quadrupole_moment = {
                        "raw_au": raw_au,
                        "raw_buckingham": raw_buck,
                        "diagonal_au": diag_au,
                        "diagonal_buckingham": diag_buck,
                        "traceless_buckingham": traceless_buck,
                        "orientation": orientation,
                        "isotropic": isotropic,
                    }
                except Exception as exc:
                    logger.warning("Failed to parse QUADRUPOLE MOMENT section: %s", exc)

            elif "CARTESIAN COORDINATES (ANGSTROEM)" in line:
                fout.readline()
                line = fout.readline()
                species = list()
                coords = list()
                while m := cart_patt.match(line):
                    species.append(m.group(1))
                    coords.append([float(val) for val in m.group(2, 3, 4)])
                    line = fout.readline()
                # change dymmy spaces with X
                if "-" in species:
                    for i, specie in enumerate(species):
                        if specie == "-":
                            species[i] = "X"
                # set up molecule object
                if self.charge and self.spin_multiplicity:
                    self.structures.append(
                        PymatgenMolecule(
                            species,
                            coords,
                            charge=self.charge,
                            spin_multiplicity=self.spin_multiplicity,
                        )
                    )
                else:
                    self.structures.append(PymatgenMolecule(species, coords))

            # ---------------------------------------------------
            # STATIC POLARIZABILITY TENSOR (Dipole/Dipole)
            # ---------------------------------------------------
            elif "STATIC POLARIZABILITY TENSOR (Dipole/Dipole)" in line:
                try:
                    # Find the raw cartesian tensor
                    while line and "The raw cartesian tensor" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    raw = None
                    if line and "The raw cartesian tensor" in line:
                        rows = []
                        for _ in range(3):
                            line = fout.readline()
                            if not line:
                                break
                            vals = [float(v) for v in float_patt.findall(line)[:3]]
                            rows.append(vals)
                        if len(rows) == 3 and all(len(r) == 3 for r in rows):
                            raw = np.array(rows)

                    # Diagonalized tensor
                    while line and "diagonalized tensor" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    diag = None
                    if line and "diagonalized tensor" in line:
                        line = fout.readline()
                        vals = [float(v) for v in float_patt.findall(line)[:3]]
                        if len(vals) == 3:
                            diag = np.array(vals)

                    # Orientation matrix (3x3)
                    orientation = None
                    while line and "Orientation" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    if line and "Orientation" in line:
                        rows = []
                        for _ in range(3):
                            line = fout.readline()
                            if not line:
                                break
                            vals = [float(v) for v in float_patt.findall(line)[:3]]
                            rows.append(vals)
                        if len(rows) == 3 and all(len(r) == 3 for r in rows):
                            orientation = np.array(rows)

                    # Isotropic polarizability
                    isotropic = None
                    while line and "Isotropic polarizability" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    if line and "Isotropic polarizability" in line:
                        vals = float_patt.findall(line)
                        if vals:
                            isotropic = float(vals[-1])

                    self._polarizability_dipole = {
                        "raw": raw,
                        "diagonal": diag,
                        "orientation": orientation,
                        "isotropic": isotropic,
                    }
                except Exception as exc:
                    logger.warning(
                        "Failed to parse STATIC POLARIZABILITY TENSOR (Dipole/Dipole): %s",
                        exc,
                    )

            # ---------------------------------------------------
            # STATIC POLARIZABILITY TENSOR (Dipole/Quadrupole)
            # ---------------------------------------------------
            elif "STATIC POLARIZABILITY TENSOR (Dipole/Quadrupole)" in line:
                try:
                    # Find the raw cartesian tensor
                    while line and "The raw cartesian tensor" not in line:
                        line = fout.readline()
                        if not line:
                            break

                    # Component ordering for quadrupole parts
                    comp_order = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
                    comp_index = {
                        ("X", "X"): 0,
                        ("Y", "Y"): 1,
                        ("Z", "Z"): 2,
                        ("X", "Y"): 3,
                        ("Y", "X"): 3,
                        ("X", "Z"): 4,
                        ("Z", "X"): 4,
                        ("Y", "Z"): 5,
                        ("Z", "Y"): 5,
                    }
                    axis_index = {"X": 0, "Y": 1, "Z": 2}
                    raw = np.zeros((3, 6))

                    dpq_patt = re.compile(
                        r"^\s*([XYZ])-\s+([XYZ])\s+([XYZ])\s*:\s*([+-]?\d+\.\d+)")

                    while True:
                        line = fout.readline()
                        if not line or not line.strip():
                            break
                        m = dpq_patt.match(line)
                        if not m:
                            continue
                        d, q1, q2, val = m.groups()
                        i = axis_index[d]
                        j = comp_index.get((q1, q2))
                        if j is None:
                            continue
                        raw[i, j] = float(val)

                    self._polarizability_dipole_quadrupole = {
                        "raw": raw,
                        "component_order": comp_order,
                    }
                except Exception as exc:
                    logger.warning(
                        "Failed to parse STATIC POLARIZABILITY TENSOR (Dipole/Quadrupole): %s",
                        exc,
                    )

            # -------------------------------------------------------------
            # STATIC TRACELESS POLARIZABILITY TENSOR (Dipole/Quadrupole)
            # -------------------------------------------------------------
            elif (
                "STATIC TRACELESS POLARIZABILITY TENSOR (Dipole/Quadrupole)"
                in line
            ):
                try:
                    while line and "The raw cartesian tensor" not in line:
                        line = fout.readline()
                        if not line:
                            break

                    comp_order = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
                    comp_index = {
                        ("X", "X"): 0,
                        ("Y", "Y"): 1,
                        ("Z", "Z"): 2,
                        ("X", "Y"): 3,
                        ("Y", "X"): 3,
                        ("X", "Z"): 4,
                        ("Z", "X"): 4,
                        ("Y", "Z"): 5,
                        ("Z", "Y"): 5,
                    }
                    axis_index = {"X": 0, "Y": 1, "Z": 2}
                    raw = np.zeros((3, 6))

                    dpq_patt = re.compile(
                        r"^\s*([XYZ])-\s+([XYZ])\s+([XYZ])\s*:\s*([+-]?\d+\.\d+)")

                    while True:
                        line = fout.readline()
                        if not line or not line.strip():
                            break
                        m = dpq_patt.match(line)
                        if not m:
                            continue
                        d, q1, q2, val = m.groups()
                        i = axis_index[d]
                        j = comp_index.get((q1, q2))
                        if j is None:
                            continue
                        raw[i, j] = float(val)

                    self._polarizability_dipole_quadrupole_traceless = {
                        "raw": raw,
                        "component_order": comp_order,
                    }
                except Exception as exc:
                    logger.warning(
                        "Failed to parse STATIC TRACELESS POLARIZABILITY TENSOR (Dipole/Quadrupole): %s",
                        exc,
                    )

            # ---------------------------------------------------
            # STATIC POLARIZABILITY TENSOR (Velocity)
            # ---------------------------------------------------
            elif "STATIC POLARIZABILITY TENSOR (Velocity)" in line:
                try:
                    # Find the raw cartesian tensor
                    while line and "The raw cartesian tensor" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    raw = None
                    if line and "The raw cartesian tensor" in line:
                        rows = []
                        for _ in range(3):
                            line = fout.readline()
                            if not line:
                                break
                            vals = [float(v) for v in float_patt.findall(line)[:3]]
                            rows.append(vals)
                        if len(rows) == 3 and all(len(r) == 3 for r in rows):
                            raw = np.array(rows)

                    # Diagonalized tensor
                    while line and "diagonalized tensor" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    diag = None
                    if line and "diagonalized tensor" in line:
                        line = fout.readline()
                        vals = [float(v) for v in float_patt.findall(line)[:3]]
                        if len(vals) == 3:
                            diag = np.array(vals)

                    # Orientation matrix (3x3)
                    orientation = None
                    while line and "Orientation" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    if line and "Orientation" in line:
                        rows = []
                        for _ in range(3):
                            line = fout.readline()
                            if not line:
                                break
                            vals = [float(v) for v in float_patt.findall(line)[:3]]
                            rows.append(vals)
                        if len(rows) == 3 and all(len(r) == 3 for r in rows):
                            orientation = np.array(rows)

                    # Isotropic polarizability
                    isotropic = None
                    while line and "Isotropic polarizability" not in line:
                        line = fout.readline()
                        if not line:
                            break
                    if line and "Isotropic polarizability" in line:
                        vals = float_patt.findall(line)
                        if vals:
                            isotropic = float(vals[-1])

                    self._polarizability_velocity = {
                        "raw": raw,
                        "diagonal": diag,
                        "orientation": orientation,
                        "isotropic": isotropic,
                    }
                except Exception as exc:
                    logger.warning(
                        "Failed to parse STATIC POLARIZABILITY TENSOR (Velocity): %s",
                        exc,
                    )

            # ---------------------------------------------------
            # STATIC HYPERPOLARIZABILITY TENSOR
            # ---------------------------------------------------
            elif "STATIC HYPERPOLARIZABILITY TENSOR" in line:
                """Parse the static hyperpolarizability section into a dict.

                The expected ORCA output looks like::

                    STATIC HYPERPOLARIZABILITY TENSOR

                      The raw Cartesian tensor (atomic units):

                         ( x x x ):          -0.000018
                         ( x x y ):          -0.000000
                         ...

                    Hyperpolarizability calculation done

                We read from the heading line until the terminating
                "Hyperpolarizability calculation done" line. For each
                non-empty line in between, we split at ":" and use the
                left-hand part to build ijk keys like "xxx", "xxy", ...,
                and the right-hand side as the numeric value. The parsed
                data are stored directly as a flat dict::

                    self._hyperpolarizability = {"xxx": -0.000018, ...}

                This avoids any axis-index conversion here; downstream
                consumers work directly with the ijk-keyed mapping.
                """

                try:
                    hyperpol = {}
                    reading = True

                    while reading:
                        line = fout.readline()
                        if not line:
                            # EOF
                            break

                        # End of hyperpolarizability block
                        if "Hyperpolarizability calculation done" in line:
                            reading = False
                            break

                        # Skip empty / whitespace-only lines
                        if not line.strip():
                            continue

                        # We only care about lines that contain a ':'
                        if ":" not in line:
                            continue
                        if any(k in line for k in ("Method", "Type", "Multiplicity", "Irrep", "Basis", "raw")):
                            # Skip lines that contain metadata about the calculation
                            continue

                        index_part, value_part = line.split(":", 1)

                        # Clean "( x x x )" -> "xxx"
                        key_part = index_part.replace("(", "").replace(")", "").strip()
                        ijk = key_part.replace(" ", "").lower()

                        if False: # can be set to True for debugging
                            # Defensive: ensure ijk looks like three characters
                            if len(ijk) != 3:
                                logger.info(
                                    "Skipping malformed hyperpolarizability index '%s' from line: %s",
                                    ijk,
                                    line.strip(),
                                )
                                continue

                        # Extract the numeric value from the right-hand side
                        try:
                            value_str = value_part.strip().split()[0]
                            value = float(value_str)
                        except (IndexError, ValueError) as exc_val:
                            logger.info(
                                "Failed to parse hyperpolarizability value from line '%s': %s",
                                line.strip(),
                                exc_val,
                            )
                            continue

                        hyperpol[ijk] = value

                    # Store the flat ijk->value mapping directly
                    self._hyperpolarizability = hyperpol

                except Exception as exc:
                    logger.warning(
                        "Failed to parse STATIC HYPERPOLARIZABILITY TENSOR: %s",
                        exc,
                    )

            # elif "THERMOCHEMISTRY AT " in line:
            #     self.thermochemistry = True
            #     self.thermo_data = dict()
            #     self.frequencies = list()

            elif "THE OPTIMIZATION HAS CONVERGED" in line:
                self.optimization_converged = True

            elif "****ORCA TERMINATED NORMALLY****" in line:
                self.normal_termination = True

            elif self.thermochemistry:
                if "Temperature" in line:
                    self.temperature = float(line.split()[-2])
                elif "Pressure" in line:
                    self.pressure = float(line.split()[-2])
                elif "Total Mass" in line:
                    self.mass = float(line.split()[-2])
                elif m := thermo_patt.match(line):
                    gdict = m.groupdict()
                    name = gdict["_name"].strip().lower()
                    uaval = float(gdict["uaval"])
                    kcalval = (
                        float(gdict["kcalval"])
                        if gdict["kcalval"]
                        else uaval * UA_TO_KCAL
                    )
                    self.thermo_data[name] = (uaval, kcalval)
                elif m := freq_patt.match(line):
                    self.frequencies.append(float(m.group(1)))
                elif "Total thermal energy" in line:
                    value = float(line.split()[-2])
                    self.thermo_data["inner energy"] = (value, value * UA_TO_KCAL)

                if "G-E(el)" in line:
                    # remove temporary thermochemistry read
                    self.thermochemistry = False

        # clean up thermochemistry data
        if self.thermo_data:
            self.thermochemistry = True
            self.thermo_data.pop("total free energy")  # this is inner energy
            self.thermo_data["entropy"] = self.thermo_data.pop("final entropy term")

        # set up an OrcaInput instance with the LAST geometry
        self.orca_input = OrcaInput(
            self.final_structure,
            charge=self.charge,
            spin_multiplicity=self.spin_multiplicity,
            input_parameters=input_parameters,
            blocks=[blocks],
        )


class OrcaEnGradfile:
    """
    Parser for ORCA EnGrad file.

    .. attribute:: filename

        Path to the ORCA engrad file

    .. attribute:: grad

        Gradient of the molecule in units of Hartree/Bohr

    """

    def __init__(self, filename):
        """Reads an ORCA EnGrad file

        Args:
            filename: Filename of Orca engrad file.
        """
        self.filename = filename

        # initialisation of all attributes, thus they will always exist
        self.grad = list()

        # parse file
        self._parse()

    @property
    def gradient(self):
        """gradient"""
        return self.grad

    def _parse(self):
        """Parse the file and fill in object and attributes"""

        with open(self.filename, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                #
                # Read input file
                #
                if "# The current gradient in Eh/bohr" in line:
                    line = f.readline()
                    line = f.readline()

                    while "#" not in line:
                        self.grad.append(float(line.rstrip()))
                        line = f.readline()

            self.grad = np.array(self.grad).reshape(int(len(self.grad) / 3), 3)


class OrcaOutput(OrcaOutfile):
    """A class to gather all results from an Orca ??? VERSION??? calculations. When
    instanciated the class will try to read all available files according
    to the given basename. The class inherit from the OrcaOutfile, the
    class includes thus all the attributes of the OrcaOutfile class.

    .. attribute: basename

        basename of Orca files of current job.

    """

    def __init__(self, basename, node_runner=None):
        """The class looks for all files corresponding to the basename
        and instanciate the corresponding object.

        Args:
            basename (str): basename of Orca output files of the current
                job.
            node_runner (optional): A node runner instance for logging.
        """
        self.basename = basename
        self.node_runner = node_runner

        # read output file if it exists
        outfile = pathlib.Path(self.basename + ".out")
        if outfile.is_file():
            super().__init__(outfile, node_runner=self.node_runner)
        else:
            raise FileNotFoundError(
                "File %s.out not found." % basename + "Check your ORCA calculation."
            )

        # read hessian file if it exists
        hessian = pathlib.Path(self.basename + ".hess")
        if hessian.is_file():
            self.hessian = OrcaHessian(hessian)
        else:
            self.hessian = None

        # read engrad file if it exists
        engrad = pathlib.Path(self.basename + ".engrad")
        if engrad.is_file():
            self.engrad = OrcaEnGradfile(engrad)
        else:
            self.engrad = None



@unique
class Scan_type(Enum):
    """
    Enum type for Scan calculations. Only 'relaxed' and 'parameter'.
    Usage: Scan_type.relaxed Scan_type.parameter
    """

    relaxed = "relaxed"
    parameter = "parameter"

    @property
    def new_step(self):
        if self.name == Scan_type.relaxed.name:
            return "RELAXED SURFACE SCAN STEP"
        elif self.name == Scan_type.parameter.name:
            return "TRAJECTORY STEP"

    @property
    def final_msg(self):
        if self.name == Scan_type.relaxed.name:
            return "RELAXED SURFACE SCAN DONE"
        elif self.name == Scan_type.parameter.name:
            return "TRAJECTORY DONE"

    def __str__(self):
        return str(self.value)


class OrcaScanCalculation:
    """This class aims to read the standard output of an ORCA VERSION????
    calculation in case of a relaxed scan or a trajectory scan."""

    def __init__(self, filename, node_runner=None):
        """The class reads the ORCA output file and store the data in
        the attributes' instance.

        Args:
            filename: Filename or file object of an Orca output file.
            node_runner (optional): A node runner instance for logging.
        """
        # set up attributes
        self.node_runner = node_runner
        self.scan_type = ""
        self.scan_steps = None
        self.scan_coords = None
        self.normal_termination = False

        # parse file
        try:
            # try to open file
            with open(filename, "r", encoding="utf-8", errors="replace") as fout:
                self._parse(fout)
        except TypeError:
            # parse the file object
            self._parse(filename)

    def error(self, msg):
        """Log an error message"""
        if self.node_runner:
            self.node_runner.error(msg)
        else:
            logger.error(msg)

    def info(self, msg):
        """Log an info message"""
        if self.node_runner:
            self.node_runner.info(msg)
        else:
            logger.info(msg)

    def debug(self, msg):
        """Log a debug message"""
        if self.node_runner:
            self.node_runner.debug(msg)
        else:
            logger.debug(msg)

    def warning(self, msg):
        """Log a warning message"""
        if self.node_runner:
            self.node_runner.warning(msg)
        else:
            logger.warning(msg)

    def _parse(self, fout):
        """Parse the file and fill in object and attributes"""

        # define regex patterns
        coord_patt = re.compile(r"\s+(\w+)\s+(\(((\s*\d+),?){2,4}\))?\s+:\s+(\d+.\d+)")
        scf_cvg_patt = re.compile(r"^\s+\*\s+SCF CONVERGED AFTER\s+(\d+)\s+CYCLES")

        # look for calculation starts and read header of the file
        calculation_start = False
        head = ""
        for line in fout:
            head += line
            if "**END OF INPUT**" in line:
                calculation_start = True
                break

        if not calculation_start:
            return None

        # look for the begining of the scan in the next 10 lines
        n = 0
        scan = False
        while n < 10 or not scan:
            line = fout.readline()
            n += 1
            if "Relaxed Surface Scan" in line:
                self.scan_type = Scan_type.relaxed
                scan = True
                break
            if "Parameter Scan Calculation" in line:
                self.scan_type = Scan_type.parameter
                scan = True

        if not scan:
            raise ValueError("This file does not correspond to " "a scan calculation.")

        self.scan_steps = list()
        self.coordinates = list()
        # look for the first step
        while self.scan_type.new_step not in line:
            line = fout.readline()

        # read coordinates value of the scan
        step_coords = dict()
        while "**********" not in line:
            line = fout.readline()
            if m := coord_patt.search(line):
                if self.scan_type == Scan_type.relaxed:
                    s = m.group(2)
                    coord = tuple(int(i) for i in s.strip(")(").split(","))
                elif self.scan_type == Scan_type.parameter:
                    coord = m.group(1)

                value = float(m.groups()[-1])
                step_coords[coord] = value

        self.scan_coords = [step_coords]

        # split the file in independant parts
        # each part is an independant calculation on a step.
        # instantiate an OrcaOutfile object for each part
        scf_cvg = False
        opt_cvg = False
        part = head + line
        end_file = False
        while self.scan_type.final_msg not in line and not end_file:
            line = fout.readline()

            # check end file
            if line == "":
                end_file = True
                continue

            if self.scan_type.new_step in line:
                # a new step will begin. Save the current one and
                # initialize variables for the next one.
                if scf_cvg:
                    if self.scan_type == Scan_type.relaxed:
                        # warning: here only the last scf step converge
                        if opt_cvg:
                            part += 30 * " " + "****ORCA TERMINATED NORMALLY****"
                    else:
                        part += 30 * " " + "****ORCA TERMINATED NORMALLY****"

                out = OrcaOutfile(io.StringIO(part))
                self.scan_steps.append(out)

                # start a new independant step
                part = head + line
                scf_cvg = False
                opt_cvg = False

                # read coordinates value of the scan
                step_coords = dict()
                while "**********" not in line:
                    line = fout.readline()
                    if m := coord_patt.search(line):
                        if self.scan_type == Scan_type.relaxed:
                            s = m.group(2)
                            coord = tuple(int(i) for i in s.strip(")(").split(","))
                        elif self.scan_type == Scan_type.parameter:
                            coord = m.group(1)

                        value = float(m.groups()[-1])
                        step_coords[coord] = value

                self.scan_coords.append(step_coords)
            else:
                part += line
                if scf_cvg_patt.match(line):
                    scf_cvg = True
                elif "THE OPTIMIZATION HAS CONVERGED" in line:
                    opt_cvg = True

        # add last step
        if scf_cvg:
            if self.scan_type == Scan_type.relaxed:
                # warning: here only the last scf step converge
                if opt_cvg:
                    part += 30 * " " + "****ORCA TERMINATED NORMALLY****"
            else:
                part += 30 * " " + "****ORCA TERMINATED NORMALLY****"

        out = OrcaOutfile(io.StringIO(part))
        self.scan_steps.append(out)

        # read the end of the output
        for line in fout:
            if "****ORCA TERMINATED NORMALLY****" in line:
                self.normal_termination = True

    def get_surface(self, dataframe=True):
        """Return the surface scan results

        Args:
            dataframe (bool): if True (default) return the surface as
                a pandas DataFrame

        """
        df = pd.DataFrame(self.scan_coords)
        df["energy"] = [step.final_energy for step in self.scan_steps]

        if dataframe:
            return df
        else:
            return df.values

    @property
    def structures(self):
        """List of final structures corresponding to each step of the
        scan calculation as pymatgen.Molecule object. In case of
        relaxed scan calculations it corresponds to the optimized
        geometry."""

        return [step.final_structure for step in self.scan_steps]

    @property
    def all_structures(self):
        """Return all structures in the calculation. For relaxed scan
        it means intermediate non-optimized structures."""
        return [s for step in self.scan_steps for s in step.structures]

    @property
    def energies(self):
        """FINAL SINGLE POINT ENERGY of each scan step"""
        return [step.final_energy for step in self.scan_steps]

    @property
    def ndim(self):
        """dimension of the energy surface"""
        return len(self.scan_coords[0])

    def export_xyz_trj(self, filename):
        """Export the structures of each scan step in a trajectory
        file in xyz format.

        Args:
            filename (string, file): path or file object
        """
        # set up trajectory file
        xyz = ""
        for structure in self.structures:
            xyz += structure.to("xyz") + "\n"

        try:
            # try to open the file and write trajectory
            with open(filename, "w", encoding="utf-8") as fout:
                fout.write(xyz)
        except TypeError:
            # write trajectory to the file
            fout.write(xyz)

    def __getitem__(self, i):
        return self.scan_steps[i]

    def __len__(self):
        return len(self.scan_steps)
