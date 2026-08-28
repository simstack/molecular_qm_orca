from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_models.dispersion_correction import (
    DispersionCorrection as SharedDispersionCorrection,
    DispersionCorrectionEnum,
)
from molecular_qm_models.molecule import Atom, Molecule
from ..lib.orca_main_lib import set_method_and_basis_set_for_non_casscf_methods
from ..models.orca_functional import OrcaFunctional
from ..models.orca_qm_input import OrcaDispersionCorrection, OrcaQMInput
from molecular_qm_models.qm_input import QMMethod


def _water() -> Molecule:
    molecule = Molecule()
    molecule.add_atom(Atom.from_coords("O", [0.0, 0.0, 0.117]))
    molecule.add_atom(Atom.from_coords("H", [0.0, 0.755, -0.471]))
    molecule.add_atom(Atom.from_coords("H", [0.0, -0.755, -0.471]))
    return molecule


def _qm_input(**overrides) -> OrcaQMInput:
    payload = {
        "molecule": _water(),
        "method": QMMethod.DFT,
        "functional": FunctionalEnum.B3LYP,
        "dispersion_correction": OrcaDispersionCorrection(value=DispersionCorrectionEnum.NONE),
    }
    payload.update(overrides)
    return OrcaQMInput(**payload)


def test_orca_functional_has_no_nested_dispersion():
    assert "dispersion_correction" not in OrcaFunctional.model_fields
    assert "dispersion_correction" in OrcaQMInput.model_fields
    assert "functional" in OrcaQMInput.model_fields


def test_orca_qm_input_schema_keeps_dispersion_as_sibling():
    schema = OrcaQMInput.json_schema()
    ui = OrcaQMInput.ui_schema()
    functional_schema = schema["properties"]["functional"]
    assert "dispersion_correction" in schema["properties"]
    assert "dispersion_correction" not in functional_schema.get("properties", {})
    assert "anyOf" not in schema["properties"]["dispersion_correction"]
    assert "basis_set" in schema["required"]
    assert "functional" in schema["required"]
    assert "dispersion_correction" in schema["required"]
    assert ui["ui:order"].index("functional") < ui["ui:order"].index("dispersion_correction")


def test_nested_functional_dispersion_is_lifted():
    nested = SharedDispersionCorrection(value=DispersionCorrectionEnum.D3BJ)
    model = _qm_input(
        functional=Functional(functional=FunctionalEnum.PBE, dispersion_correction=nested),
        dispersion_correction=None,
    )
    assert model.functional.keyword() == "PBE"
    assert "dispersion_correction" not in type(model.functional).model_fields
    assert model.dispersion_correction.value == DispersionCorrectionEnum.D3BJ


def test_dft_first_line_uses_sibling_dispersion():
    qm_input = _qm_input(
        functional=FunctionalEnum.B3LYP,
        dispersion_correction=OrcaDispersionCorrection(value=DispersionCorrectionEnum.D3BJ),
    )
    first_line = set_method_and_basis_set_for_non_casscf_methods(None, qm_input, "")
    assert "B3LYP" in first_line
    assert "D3BJ" in first_line
    assert "DFT-D3BJ" not in first_line
