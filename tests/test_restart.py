import pytest
import os
from pathlib import Path
from simstack.core.definitions import TaskStatus
from simstack.models import Parameters
from simstack.models.files import FileStack
from molecular_qm_models import QMInput, BasisSet, QMMethod
from molecular_qm_models.basis_set import BasisSetEnum
from molecular_qm_models.density_functional import Functional, FunctionalEnum
from molecular_qm_models.dispersion_correction import DispersionCorrection, DispersionCorrectionEnum
from molecular_qm_orca.orca import orca
from simstack.models.file_list import FileList
from .fixtures import water

@pytest.mark.asyncio
async def test_orca_restart(initialized_context, water, tmp_path):
    """
    Test ORCA restart functionality.
    1. Run a calculation with very few SCF iterations, expect failure.
    2. Restart from the generated GBW file with more iterations, expect success.
    """
    
    # 1. Setup first run: limited SCF iterations
    basis = BasisSet(basis_set=BasisSetEnum.Def2_TZVP)
    disp = DispersionCorrection(value=DispersionCorrectionEnum.NONE)
    functional = Functional(functional=FunctionalEnum.B3LYP, dispersion_correction=disp)

    qm_input_fail = QMInput(
        molecule=water,
        method=QMMethod.DFT,
        basis_set=basis,
        functional=functional,
        charge=0,
        multiplicity=1,
        max_scf_iterations=2,
        tolerate_failure=True  # Important to get results even if SCF fails
    )
    
    parameters = Parameters(resource="local", force_rerun=True)
    
    print("\n--- Starting First Run (Expect SCF failure) ---")
    simstack_result_fail = await orca(qm_input_fail, parameters=parameters)
    
    assert simstack_result_fail.status == TaskStatus.COMPLETED
    qm_result_fail = simstack_result_fail.orca_result
    
    # Check that it didn't converge as expected
    assert qm_result_fail.scf_converged is False
    assert qm_result_fail.normal_termination is False
    
    # 2. Get the GBW file for restart
    gbw_file = next((f for f in qm_result_fail.files if f.name == "orca.gbw"), None)
    assert gbw_file is not None, "orca.gbw should be in qm_result.files"
    
    # 3. Setup second run: restart from GBW and more iterations
    restart_files = FileList()
    restart_files.append(gbw_file)
    
    qm_input_restart = QMInput(
        molecule=water,
        method=QMMethod.DFT,
        basis_set=basis,
        functional=functional,
        charge=0,
        multiplicity=1,
        max_scf_iterations=20,
        restart_files=restart_files
    )
    
    print("\n--- Starting Second Run (Restart from GBW, expect success) ---")
    simstack_result_success = await orca(qm_input_restart, parameters=parameters)
    
    assert simstack_result_success.status == TaskStatus.COMPLETED
    qm_result_success = simstack_result_success.orca_result
    
    assert qm_result_success.scf_converged is True
    assert qm_result_success.normal_termination is True
    assert qm_result_success.final_energy is not None
    
    # Verify that the restart was actually used in the ORCA input
    inp_file = next((f for f in simstack_result_success.info_files if f.name == "orca.inp"), None)
    if inp_file:
        inp_path = inp_file.get(tmp_path)
        content = inp_path.read_text()
        assert "MORead" in content
        assert '%moinp "orca.gbw"' in content
