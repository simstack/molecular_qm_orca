import pytest
import os
import json
from pathlib import Path
from simstack.core.definitions import TaskStatus
from odmantic import ObjectId
from molecular_qm_models import BasisSet
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_models.density_functional import FunctionalEnum
from molecular_qm_models.dispersion_correction import DispersionCorrectionEnum
from molecular_qm_models.qm_input import QMMethod, SolventModel
from molecular_qm_orca.models import OrcaDispersionCorrection, OrcaFunctional, OrcaQMInput
from molecular_qm_orca.orca import orca
from simstack.models import Parameters
from .fixtures import water

def approx_compare(obj1, obj2, rel=1e-6, abs=1e-6, path=""):
    """Recursively compare two objects with approximate matching for floats."""
    if isinstance(obj1, float) and isinstance(obj2, float):
        if not (pytest.approx(obj1, rel=rel, abs=abs) == obj2):
            print(f"Float mismatch at {path}: {obj1} != {obj2}")
            return False
        return True
    elif isinstance(obj1, dict) and isinstance(obj2, dict):
        if len(obj1) != len(obj2):
            print(f"Dict length mismatch at {path}: {len(obj1)} != {len(obj2)}")
            print(f"Keys in 1: {set(obj1.keys())}")
            print(f"Keys in 2: {set(obj2.keys())}")
            return False
        for k, v in obj1.items():
            if k not in obj2:
                print(f"Key {k} missing at {path}")
                return False
            if not approx_compare(v, obj2[k], rel, abs, f"{path}.{k}" if path else k):
                return False
        return True
    elif isinstance(obj1, list) and isinstance(obj2, list):
        if len(obj1) != len(obj2):
            print(f"List length mismatch at {path}: {len(obj1)} != {len(obj2)}")
            return False
        for i in range(len(obj1)):
            if not approx_compare(obj1[i], obj2[i], rel, abs, f"{path}[{i}]"):
                return False
        return True
    else:
        if obj1 != obj2:
            print(f"Value mismatch at {path}: {obj1} ({type(obj1)}) != {obj2} ({type(obj2)})")
            return False
        return True

@pytest.mark.asyncio
async def test_water_dft(initialized_context, water, tmp_path, gather):
    """
    Test DFT calculations with various functionals, basis sets, and dispersion corrections.
    Also tests multiple solvents.
    """
    functionals = [FunctionalEnum.B3LYP, FunctionalEnum.PBE]
    basis_sets = [BasisSetEnum.Def2_SVP, BasisSetEnum.Def2_TZVP]
    dispersion_options = [DispersionCorrectionEnum.NONE, DispersionCorrectionEnum.D3BJ]

    test_cases = []
    # DFT combinations
    for f_enum in functionals:
        for b_enum in basis_sets:
            for d_enum in dispersion_options:
                test_cases.append({
                    "name": f"water_{f_enum}_{b_enum}_{d_enum}".lower().replace("-", "_"),
                    "functional": f_enum,
                    "basis": b_enum,
                    "dispersion": d_enum,
                    "solvent": "None",
                    "solvent_model": SolventModel.CPCM
                })
    
    # Multiple solvents with one basis set (Def2-SVP, B3LYP, NONE)
    solvents = ["Ethanol", "Water"]
    for solvent in solvents:
        test_cases.append({
            "name": f"water_b3lyp_def2_svp_none_{solvent.lower()}".replace("-", "_"),
            "functional": FunctionalEnum.B3LYP,
            "basis": BasisSetEnum.Def2_SVP,
            "dispersion": DispersionCorrectionEnum.NONE,
            "solvent": solvent,
            "solvent_model": SolventModel.CPCM
        })

    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    for case in test_cases:
        basis = BasisSet(basis_set=case["basis"])
        qm_input = OrcaQMInput(
            molecule=water,
            method=QMMethod.DFT,
            basis_set=basis,
            functional=OrcaFunctional(functional=case["functional"]),
            dispersion_correction=OrcaDispersionCorrection(value=case["dispersion"]),
            charge=0,
            multiplicity=1,
            solvent=case["solvent"],
            use_solvent=(case["solvent"] != "None"),
            solvent_model=case["solvent_model"]
        )
        
        parameters = Parameters(resource="local", force_rerun=True)
        simstack_result = await orca(qm_input, parameters=parameters)
        
        assert simstack_result.status == TaskStatus.COMPLETED, f"Node failed for {case['name']}: {simstack_result.error_message}"
        assert hasattr(simstack_result, "orca_result")
        qm_result = simstack_result.orca_result

        # Call custom_model_dump
        current_dump = await qm_result.custom_model_dump()
        
        # Clean current_dump from ObjectId and other non-JSON serializable if any
        # Also remove unstable fields like 'files', 'info_files', and 'id'
        class SimstackJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if obj.__class__.__name__ == 'ObjectId':
                    return str(obj)
                # handle potential Pydantic objects or others that have model_dump
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                return super().default(obj)

        def clean_dump(obj):
            if obj.__class__.__name__ == 'ObjectId':
                return str(obj)
            if isinstance(obj, dict):
                # Remove unstable fields from QMResult or sub-models
                # Also remove 'elements' from structures as it seems to contain IDs in some versions
                res = {}
                for k, v in obj.items():
                    if k in ['files', 'info_files', 'id', 'task_id', 'elements']:
                        continue
                    res[k] = clean_dump(v)
                return res
            if isinstance(obj, list):
                return [clean_dump(i) for i in obj]
            return obj
        
        current_dump = clean_dump(current_dump)
        
        file_path = data_dir / f"{case['name']}.json"

        if gather:
            with open(file_path, "w") as f:
                json.dump(current_dump, f, indent=2, cls=SimstackJSONEncoder)
            print(f"Stored results for {case['name']} in {file_path}")
        else:
            assert file_path.exists(), f"Stored data not found for {case['name']} at {file_path}. Run with --gather first."
            with open(file_path, "r") as f:
                stored_dump = json.load(f)
            
            # Compare dumps
            # We skip comparing files and info_files content since they might contain paths or IDs
            # but QMResult usually contains energies and structures which are important.
            # custom_model_dump of QMResult includes final_energy, dipole, dipole_moment, etc.
            
            # We use a helper for approximate comparison of floats
            assert approx_compare(current_dump, stored_dump), f"Results for {case['name']} do not match stored data."
