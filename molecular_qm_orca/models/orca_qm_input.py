from typing import Any, List, Optional, Self

from odmantic import EmbeddedModel, Field, Model, Reference
from pydantic import model_validator

from molecular_qm_models.basis_set import BasisSet
from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.molecule import Molecule
from molecular_qm_models.qm_input import (
    ElProp,
    GridType,
    OptimizationAccuracy,
    QMMethod,
    SCFAccuracy,
    SolventModel,
)
from .orca_functional import (
    OrcaFunctional,
    as_orca_functional_doc,
)
from simstack.models import simstack_model
from simstack.models.file_list import FileList
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


def _nested_dispersion_payload(functional: Any) -> Any:
    if isinstance(functional, dict):
        nested = functional.get("dispersion_correction")
    else:
        nested = getattr(functional, "dispersion_correction", None)
    if nested in (None, {}):
        return None
    if hasattr(nested, "model_dump"):
        return nested.model_dump(exclude={"id"})
    return nested


def normalize_functional_and_dispersion(data: dict) -> dict:
    """Lift nested functional.dispersion_correction onto the sibling field."""
    functional = data.get("functional")
    if data.get("dispersion_correction") in (None, {}):
        nested = _nested_dispersion_payload(functional)
        if nested is not None:
            data["dispersion_correction"] = nested
        else:
            data.pop("dispersion_correction", None)
    if functional is not None:
        if isinstance(functional, OrcaFunctional):
            data["functional"] = functional
        else:
            data["functional"] = as_orca_functional_doc(functional)
    return data


@simstack_model
class OrcaDispersionCorrection(EmbeddedModel):
    """Dispersion correction for ORCA jobs, including non-DFT methods."""

    field_name: str = "DispersionCorrection"
    value: DispersionCorrectionEnum = Field(
        default=DispersionCorrectionEnum.NONE,
        json_schema_extra={
            "enum": [item.value for item in DispersionCorrectionEnum],
            "description": "Version of the dispersion correction to use",
            "title": "Dispersion Correction",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            if data in (None, ""):
                return {"field_name": cls.__name__, "value": DispersionCorrectionEnum.NONE}
            return {"field_name": cls.__name__, "value": data}
        data.pop("id", None)
        data.pop("_id", None)
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        return data

    def keyword(self) -> str:
        value = self.value
        raw = value.value if isinstance(value, DispersionCorrectionEnum) else str(value)
        if not raw or raw.upper() == DispersionCorrectionEnum.NONE.value:
            return ""
        return raw

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        schema["description"] = "Parameters for dispersion corrections"
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        return ui_schema


@simstack_model
class OrcaQMInput(Model):
    """ORCA-specific quantum-mechanical input.

    Field surface follows the shared QM settings used by the ORCA node, but
    functional is :class:`OrcaFunctional` (no nested dispersion) and
    :class:`DispersionCorrection` is a required sibling (use NONE to omit it).
    """

    model_config = {"extra": "ignore"}
    field_name: str = "OrcaQMInput"

    molecule: Molecule = Reference()
    charge: int = Field(0, json_schema_extra={"description": "net charge of the molecule"})
    multiplicity: int = Field(1, json_schema_extra={"description": "singlet,triplet,....."})
    open_shell_calculation: bool = Field(False, json_schema_extra={"description": "Open shell calculation"})
    basis_set: BasisSet = Field(default_factory=BasisSet)
    functional: OrcaFunctional = Field(default_factory=OrcaFunctional)
    dispersion_correction: OrcaDispersionCorrection = Field(
        default_factory=OrcaDispersionCorrection,
        json_schema_extra={
            "description": (
                "Dispersion correction for the calculation. Independent of the "
                "density functional so HF, MP2, and other non-DFT jobs can set it. "
                "Use NONE to omit a correction."
            ),
            "title": "Dispersion Correction",
        },
    )
    method: QMMethod = Field(QMMethod.DFT, json_schema_extra={"description": "Quantum chemistry calculation method"})
    gradients: bool = Field(
        False, json_schema_extra={"description": "Calculate gradients (forces) for the molecule"}
    )
    optimization: bool = Field(False, json_schema_extra={"description": "Perform geometry optimization"})
    frequencies: bool = Field(False, json_schema_extra={"description": "Calculate frequencies"})

    solvent: str = "None"
    solvent_model: SolventModel = Field(SolventModel.CPCM, json_schema_extra={"description": "Solvent model to use"})

    print_level: int = Field(1, json_schema_extra={"description": "Print level for the calculation, 0-4"})
    scf_accuracy: SCFAccuracy = Field(
        SCFAccuracy.Medium, json_schema_extra={"description": "SCF convergence accuracy"}
    )
    optimization_accuracy: OptimizationAccuracy = Field(
        OptimizationAccuracy.Medium, json_schema_extra={"description": "Geometry optimization accuracy"}
    )
    grid_type: GridType = Field(GridType.Grid2, json_schema_extra={"description": "DFT grid quality level"})
    grid_spacing: float = Field(0.2, json_schema_extra={"description": "Grid spacing for the DFT calculation"})
    max_scf_iterations: int = Field(100, json_schema_extra={"description": "Maximum number of SCF iterations"})
    max_optimization_iterations: int = Field(
        100, json_schema_extra={"description": "Maximum number of geometry optimization iterations"}
    )
    first_line: Optional[str] = Field(
        "", json_schema_extra={"description": "additional input to the line of the input file"}
    )
    blocks: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "description": "Additional blocks of text to be included in the input file, e.g. for constraints or custom settings"
        },
    )

    Dipole: bool = Field(False, json_schema_extra={"description": "Calculate dipole moment"})
    Quadrupole: bool = Field(False, json_schema_extra={"description": "Calculate quadrupole moment"})
    Polar: bool = Field(False, json_schema_extra={"description": "Calculate polarizability"})
    Hyperpol: bool = Field(False, json_schema_extra={"description": "Calculate hyperpolarizability"})
    HyperpolFrequencynm: float = Field(
        0.0, json_schema_extra={"description": "Frequency for hyperpolarizability calculation in nm"}
    )
    PolarVelocity: bool = Field(
        False, json_schema_extra={"description": "Calculate polarizability using velocity gauge"}
    )
    PolarDipQuad: bool = Field(
        False, json_schema_extra={"description": "Calculate dipole-quadrupole polarizability"}
    )
    PolarQuadQuad: bool = Field(
        False, json_schema_extra={"description": "Calculate quadrupole-quadrupole polarizability"}
    )

    states: int = Field(
        0, json_schema_extra={"description": "number of states to calculate, zero for ground state only"}
    )
    focus_state: int = Field(1, json_schema_extra={"description": "state of focus"})
    active_electrons: int = Field(0, json_schema_extra={"description": "number of active electrons"})
    active_orbitals: int = Field(0, json_schema_extra={"description": "number of active orbitals"})

    restart_files: FileList = Field(
        default_factory=FileList, json_schema_extra={"description": "Files to be used for restarting the calculation"}
    )
    tolerate_failure: bool = Field(False, json_schema_extra={"description": "Tolerate failure of the calculation"})

    excited_states: bool = Field(False, json_schema_extra={"title": "Calculate excited states"})
    compute_properties: bool = Field(False, json_schema_extra={"title": "Compute properties"})
    use_solvent: bool = Field(False, json_schema_extra={"title": "Use solvent"})
    non_standard_inputs: bool = Field(False, json_schema_extra={"title": "Non-standard inputs"})
    non_standard_parameters: bool = Field(False, json_schema_extra={"title": "Non-standard parameters"})

    @model_validator(mode="before")
    @classmethod
    def validate_before(cls, data):
        if not isinstance(data, dict):
            return data
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        if "restart_files" not in data or data.get("restart_files") is None:
            data["restart_files"] = {}
        if "solvent" not in data or data["solvent"] is None:
            data["solvent"] = "None"
        if "frequencies" not in data or data["frequencies"] is None:
            data["frequencies"] = False
        if "use_solvent" not in data:
            data["use_solvent"] = data.get("solvent", "None") != "None"

        elprop_data = data.pop("elprop", {})
        if isinstance(elprop_data, dict):
            for key, value in elprop_data.items():
                if key not in data:
                    data[key] = value
        elif hasattr(elprop_data, "model_dump"):
            for key, value in elprop_data.model_dump().items():
                if key not in data:
                    data[key] = value

        es_input_data = data.pop("excited_states_input", {})
        if isinstance(es_input_data, dict):
            for key, value in es_input_data.items():
                if key not in data:
                    data[key] = value
        elif hasattr(es_input_data, "model_dump"):
            for key, value in es_input_data.model_dump().items():
                if key not in data:
                    data[key] = value

        if "excited_states" not in data:
            data["excited_states"] = data.get("states", 0) > 0

        if "compute_properties" not in data:
            elprop_fields = [
                "Dipole",
                "Quadrupole",
                "Polar",
                "Hyperpol",
                "PolarVelocity",
                "PolarDipQuad",
                "PolarQuadQuad",
            ]
            data["compute_properties"] = any(data.get(key) for key in elprop_fields) or data.get(
                "HyperpolFrequencynm", 0
            ) > 0

        if not data.get("compute_properties"):
            for field in [
                "Dipole",
                "Quadrupole",
                "Polar",
                "Hyperpol",
                "PolarVelocity",
                "PolarDipQuad",
                "PolarQuadQuad",
            ]:
                data[field] = False
            data["HyperpolFrequencynm"] = 0.0

        if not data.get("excited_states"):
            data["states"] = 0
            data["focus_state"] = 1
            data["active_electrons"] = 0
            data["active_orbitals"] = 0

        if not data.get("use_solvent"):
            data["solvent"] = "None"

        return normalize_functional_and_dispersion(data)

    @model_validator(mode="after")
    def validate_calculation_options(self) -> Self:
        if self.frequencies and self.states > 0:
            raise ValueError(
                "Frequency calculations for excited states (states > 0) are not supported. "
                "Set states=0 or frequencies=False."
            )
        if self.Hyperpol and self.states > 0:
            raise ValueError(
                "Hyperpolarizability for excited states (states > 0) is not supported. Set states=0."
            )
        if self.HyperpolFrequencynm < 0:
            raise ValueError("HyperpolFrequencynm must be >= 0 (0 = static).")
        return self

    @property
    def elprop(self) -> ElProp:
        return ElProp(
            Dipole=self.Dipole,
            Quadrupole=self.Quadrupole,
            Polar=self.Polar,
            Hyperpol=self.Hyperpol,
            HyperpolFrequencynm=self.HyperpolFrequencynm,
            PolarVelocity=self.PolarVelocity,
            PolarDipQuad=self.PolarDipQuad,
            PolarQuadQuad=self.PolarQuadQuad,
        )

    def functional_enum(self):
        return self.functional.functional

    def dispersion_enum(self) -> DispersionCorrectionEnum:
        value = self.dispersion_correction.value
        if isinstance(value, DispersionCorrectionEnum):
            return value
        return DispersionCorrectionEnum(value)

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        properties = schema.setdefault("properties", {})

        if "gradients" in properties:
            del properties["gradients"]

        if "first_line" in properties:
            first_line_schema = properties["first_line"]
            if "anyOf" in first_line_schema:
                first_line_schema["type"] = "string"
                del first_line_schema["anyOf"]

        properties.pop("functional", None)
        opt_acc_schema = properties.pop("optimization_accuracy", None)

        elprop_fields = [
            "Dipole",
            "Quadrupole",
            "Polar",
            "Hyperpol",
            "HyperpolFrequencynm",
            "PolarVelocity",
            "PolarDipQuad",
            "PolarQuadQuad",
        ]
        elprop_schemas = {field: properties.pop(field) for field in elprop_fields if field in properties}

        es_fields = ["states", "focus_state", "active_electrons", "active_orbitals"]
        es_schemas = {field: properties.pop(field) for field in es_fields if field in properties}

        nsi_fields = ["first_line", "blocks", "restart_files"]
        nsi_schemas = {field: properties.pop(field) for field in nsi_fields if field in properties}

        nsp_fields = [
            "print_level",
            "scf_accuracy",
            "grid_type",
            "grid_spacing",
            "max_scf_iterations",
            "max_optimization_iterations",
        ]
        nsp_schemas = {field: properties.pop(field) for field in nsp_fields if field in properties}

        solvent_schema = properties.pop("solvent", None)
        solvent_model_schema = properties.pop("solvent_model", None)

        properties["functional"] = OrcaFunctional.json_schema()
        properties["dispersion_correction"] = OrcaDispersionCorrection.json_schema()

        schema.setdefault("dependencies", {}).update(
            {
                "compute_properties": {
                    "oneOf": [
                        {"properties": {"compute_properties": {"const": False}}},
                        {"properties": {"compute_properties": {"const": True}, **elprop_schemas}},
                    ]
                },
                "excited_states": {
                    "oneOf": [
                        {"properties": {"excited_states": {"const": False}}},
                        {"properties": {"excited_states": {"const": True}, **es_schemas}},
                    ]
                },
                "use_solvent": {
                    "oneOf": [
                        {"properties": {"use_solvent": {"const": False}}},
                        {
                            "properties": {
                                "use_solvent": {"const": True},
                                "solvent": solvent_schema,
                                "solvent_model": solvent_model_schema,
                            }
                        },
                    ]
                },
                "non_standard_inputs": {
                    "oneOf": [
                        {"properties": {"non_standard_inputs": {"const": False}}},
                        {"properties": {"non_standard_inputs": {"const": True}, **nsi_schemas}},
                    ]
                },
                "non_standard_parameters": {
                    "oneOf": [
                        {"properties": {"non_standard_parameters": {"const": False}}},
                        {"properties": {"non_standard_parameters": {"const": True}, **nsp_schemas}},
                    ]
                },
                "optimization": {
                    "oneOf": [
                        {
                            "properties": {
                                "optimization": {"const": False},
                                "gradients": {"type": "boolean"},
                            }
                        },
                        {
                            "properties": {
                                "optimization": {"const": True},
                                "optimization_accuracy": opt_acc_schema,
                            }
                        },
                    ]
                },
            }
        )

        required = schema.setdefault("required", [])
        for name in ("basis_set", "functional", "dispersion_correction"):
            if name not in required:
                required.append(name)
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        ui_schema["non_standard_inputs"] = {"ui:widget": "checkbox", "ui:title": "Non-standard inputs"}
        ui_schema["non_standard_parameters"] = {
            "ui:widget": "checkbox",
            "ui:title": "Non-standard parameters",
        }
        ui_schema["use_solvent"] = {"ui:widget": "checkbox", "ui:title": "Use solvent"}
        ui_schema["compute_properties"] = {"ui:widget": "checkbox", "ui:title": "Compute properties"}
        ui_schema["excited_states"] = {"ui:widget": "checkbox", "ui:title": "Calculate excited states"}

        elprop_fields = [
            "Dipole",
            "Quadrupole",
            "Polar",
            "Hyperpol",
            "HyperpolFrequencynm",
            "PolarVelocity",
            "PolarDipQuad",
            "PolarQuadQuad",
        ]
        for field in elprop_fields:
            ui_schema[field] = {"ui:condition": {"compute_properties": True}}
        ui_schema["HyperpolFrequencynm"]["ui:condition"] = {
            "compute_properties": True,
            "Hyperpol": True,
        }

        for field in ["states", "focus_state", "active_orbitals", "active_electrons"]:
            ui_schema[field] = {"ui:condition": {"excited_states": True}}
        ui_schema["focus_state"]["ui:condition"] = {
            "excited_states": True,
            "method": ["CASSCF", "DFTMRCI"],
        }
        ui_schema["active_orbitals"]["ui:condition"] = {
            "excited_states": True,
            "method": ["CASSCF", "DFTMRCI"],
        }
        ui_schema["active_electrons"]["ui:condition"] = {
            "excited_states": True,
            "method": ["CASSCF", "DFTMRCI"],
        }

        ui_schema["optimization_accuracy"] = {"ui:condition": {"optimization": True}}
        ui_schema["first_line"] = {"ui:condition": {"non_standard_inputs": True}}
        ui_schema["blocks"] = {"ui:condition": {"non_standard_inputs": True}}
        ui_schema["restart_files"] = {"ui:condition": {"non_standard_inputs": True}}
        ui_schema["solvent"] = {"ui:condition": {"use_solvent": True}}
        ui_schema["solvent_model"] = {"ui:condition": {"use_solvent": True}}
        for field in [
            "print_level",
            "scf_accuracy",
            "grid_type",
            "grid_spacing",
            "max_scf_iterations",
            "max_optimization_iterations",
        ]:
            ui_schema[field] = {"ui:condition": {"non_standard_parameters": True}}

        ui_schema["ui:order"] = [
            "molecule",
            "name",
            "non_standard_inputs",
            "first_line",
            "blocks",
            "restart_files",
            "charge",
            "multiplicity",
            "open_shell_calculation",
            "method",
            "functional",
            "dispersion_correction",
            "basis_set",
            "excited_states",
            "states",
            "focus_state",
            "active_orbitals",
            "active_electrons",
            "use_solvent",
            "solvent",
            "solvent_model",
            "optimization",
            "gradients",
            "frequencies",
            "compute_properties",
            "Dipole",
            "Quadrupole",
            "Polar",
            "Hyperpol",
            "HyperpolFrequencynm",
            "PolarVelocity",
            "PolarDipQuad",
            "PolarQuadQuad",
            "non_standard_parameters",
            "print_level",
            "scf_accuracy",
            "grid_type",
            "grid_spacing",
            "max_scf_iterations",
            "max_optimization_iterations",
            "id",
            "tolerate_failure",
        ]
        ui_schema.setdefault("ui:options", {})["ui:foldable"] = True
        return ui_schema
