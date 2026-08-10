import pytest

from molecular_qm_models import Molecule
from .fixtures import water, formaldehyde

def test_water_fixture(water):
    assert len(water) == 3
    # make_formula() currently returns IUPAC name 'oxidane' for water
    assert water.make_formula() == "oxidane"

def test_formaldehyde_fixture(formaldehyde):
    assert len(formaldehyde) == 4
    # make_formula() returns 'formaldehyde'
    assert formaldehyde.make_formula() == "formaldehyde"


