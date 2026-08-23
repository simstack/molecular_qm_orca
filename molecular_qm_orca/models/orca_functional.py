from typing import Any, Dict, List

from odmantic import EmbeddedModel, Field
from pydantic import model_validator

from molecular_qm_models.density_functional import FunctionalEnum
from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema

ORCA_FUNCTIONAL_VALUES: List[str] = [item.value for item in FunctionalEnum]
ORCA_DEFAULT_FUNCTIONAL = FunctionalEnum.B3LYP


def coerce_orca_functional(raw: Any) -> FunctionalEnum:
    """Map a stored / UI / legacy name onto an ORCA functional keyword."""
    if isinstance(raw, FunctionalEnum):
        return raw
    if raw is None or raw == "":
        return ORCA_DEFAULT_FUNCTIONAL
    if isinstance(raw, dict):
        raw = raw.get("functional", raw)
    nested = getattr(raw, "functional", raw)
    if nested is not raw:
        raw = nested
    raw = getattr(raw, "value", raw)
    text = str(raw).strip()
    if not text:
        return ORCA_DEFAULT_FUNCTIONAL
    for item in FunctionalEnum:
        if item.value.lower() == text.lower() or item.name.lower() == text.lower():
            return item
    raise ValueError(f"Unsupported ORCA functional: {raw}")


def as_orca_functional_doc(raw: Any) -> Dict[str, Any]:
    functional = coerce_orca_functional(raw)
    return {
        "field_name": "OrcaFunctional",
        "functional": functional.value,
    }


@simstack_model
class OrcaFunctional(EmbeddedModel):
    """ORCA density functional. Dispersion stays a sibling on the QM input."""

    field_name: str = "OrcaFunctional"
    functional: FunctionalEnum = Field(
        ORCA_DEFAULT_FUNCTIONAL,
        json_schema_extra={
            "enum": ORCA_FUNCTIONAL_VALUES,
            "description": "ORCA density functional keyword",
            "title": "Functional",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            if isinstance(data, (FunctionalEnum, str)):
                return {
                    "field_name": cls.__name__,
                    "functional": coerce_orca_functional(data),
                }
            return data
        data.pop("id", None)
        data.pop("_id", None)
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        if "functional" in data:
            data["functional"] = coerce_orca_functional(data["functional"])
        data.pop("dispersion_correction", None)
        return data

    def keyword(self) -> str:
        value = self.functional
        if isinstance(value, FunctionalEnum):
            return value.value
        return coerce_orca_functional(value).value

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        schema["description"] = "ORCA density functional"
        properties = schema.setdefault("properties", {})
        properties["functional"] = {
            "type": "string",
            "enum": list(ORCA_FUNCTIONAL_VALUES),
            "default": ORCA_DEFAULT_FUNCTIONAL.value,
            "title": "Functional",
            "description": "ORCA density functional keyword",
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        ui_schema.setdefault("functional", {})["ui:widget"] = "select"
        return ui_schema
