import pytest

from molecular_qm_models import Molecule


@pytest.fixture
def water() -> Molecule:
    """Fixture to provide a water molecule."""
    return Molecule.from_sites(
        elements=["O", "H", "H"],
        sites=[[0.0, 0.0, 0.117], [0.0, 0.755, -0.471], [0.0, -0.755, -0.471]]
    )


@pytest.fixture
def formaldehyde() -> Molecule:
    """Fixture to provide a formaldehyde molecule."""
    return Molecule.from_sites(
        elements=["C", "O", "H", "H"],
        sites=[
            [0.0, 0.0, 0.0],  # C
            [0.0, 0.0, 1.22],  # O
            [0.0, 0.94, -0.58],  # H
            [0.0, -0.94, -0.58]  # H
        ]
    )
