import asyncio

from molecular_qm_models import Molecule, Atom
from simstack.models import Parameters
from simstack.core.context import context
import logging

logger = logging.getLogger("OrcaNode")
parameters = Parameters(resource="int-nano", queue="slurm-queue")

def multiplicity_to_string(multiplicity:int) -> str:
    table = ["singlet","doublet","triplet","quadruplet","quintuplet"]
    if multiplicity < 1 or multiplicity > 5:
        raise ValueError(f"Invalid multiplicity")
    return table[multiplicity-1]
#removed depreciated node. - IF AN EXAMPLE FOR CURRENT NODES IS NEEDED I CAN ADD IT BUT WITH THE CORRECT SPECS
# @node(parameters=parameters)
# def orca_v01(qm_input: QMInput, **kwargs) -> QMResult:
#     return orca_function_v01(qm_input, **kwargs)

# @node(parameters=parameters)
# async def async_orca_v01(qm_input: QMInput, **kwargs):
#     result = await asyncio.to_thread(orca_function_v01, qm_input, **kwargs)
#     return result


# def orca_function_v01(qm_input: QMInput, **kwargs) -> QMResult:
#     task_id = kwargs.get("task_id", "NA")
#     # generate the input files
#     logger.info("Generating input files for ORCA in task_id: {task_id}")
#     try:
#         params = [qm_input.functional.functional_str + " " + qm_input.basis_set.basis_set_str]

#         # Add this line to specify triplet state directly in the main input line

#         blocks = []

#         if qm_input.basis_set.aux_basis_:
#             params[0] += " " + qm_input.basis_set.aux_basis_.aux_basis_str
#         if qm_input.multiplicity > 1:
#             params[0] = params[0] + f" UHF"
#         if qm_input.optimization:
#             params.append("Opt")
#         elif qm_input.gradients:
#             params.append("ENGRAD")

#         # Check if there are constraints to apply
#         constraints = qm_input.molecule.properties.get("constraints", [])
#         frozen_atoms = [c["index"] + 1 for c in constraints if c["type"] == "frozen"]  # ORCA uses 1-based indexing

#         if frozen_atoms:
#             frozen_atoms_block = "%geom\n Constraints\n"
#             for atom_idx in frozen_atoms:
#                 frozen_atoms_block += f"  {{ C {atom_idx} C }}\n"  # Freeze all coordinates (x, y, z)

#             frozen_atoms_block += "end\nend\n\n"
#             blocks.append(frozen_atoms_block)

#         # blocks.append("%maxcore    1000\n%pal nprocs 8\nend\n\n")
#         with_geometry = True
#         if os.path.exists("orca.xyz"):
#             logger.info(f"OrcaInput task_id: {task_id} found orca.xyz file. Using it for geometry optimization. ")
#             with_geometry = False
#             blocks.append(f"* xyzfile {qm_input.charge} {qm_input.multiplicity} orca.xyz\n")
#         elif os.path.exists("orca.gbw"):
#             logger.info(f"OrcaInput task_id: {task_id} found orca.gbw file. Using it for geometry optimization. ")
#             blocks.append("%moinp \"orca.gbw\"\nend\n")

#         # Only add TDDFT block if states > 0
#         if qm_input.states > 0:
#             blocks.append(f"%tddft\nnroots {qm_input.states}\niroot {qm_input.focus_state}\nirootmult {multiplicity_to_string(qm_input.multiplicity)}\nend\n\n")

#         orca_input_file = OrcaInput(qm_input.molecule, charge=qm_input.charge,
#                                     spin_multiplicity=qm_input.multiplicity,
#                                     input_parameters=params, blocks=blocks, with_geometry=with_geometry)
#         print("orca_input_file: ", orca_input_file)
#         with open("orca.inp", "w") as f:
#             f.write(str(orca_input_file))

#         logger.info(f"Done generating input files for ORCA task_id: {task_id}")
#     except Exception as e:
#         logger.error(f"Error generating input files for ORCA: task_id: {task_id}", str(e))
#         return QMResult(status="input generation failed", error=str(e), task_status=TaskStatus.FAILED)

#     try:
#         command = "source ~/.bashrc && module load orca && module load openmpi && $ORCA_HOME/orca orca.inp > orca.out"
#         process = subprocess.run(
#             command,
#             shell=True,  # Important: use shell=True for shell operators like &&
#             capture_output=True,
#             text=True
#         )
#         logger.info(f"Running ORCA calculation task_id: {task_id}")
#     except Exception as e:
#         logger.error(f"Error running ORCA calculation: task_id: {task_id} {str(e)}")
#         return QMResult(status="execution failed", error=str(e), task_status=TaskStatus.FAILED)

#     result_dict = {}
#     try:

#         if os.path.exists("orca.out"):
#             # get the filename without the extension
#             path_without_extension = ""  # os.path.splitext(os.path.basename("file_path"))[0]
#             orca_run = OrcaOutput(path_without_extension)
#             if hasattr(orca_run, "scf_converge"):
#                 orca_run.success = orca_run.scf_converge
#             # if orca_run.engrad is not None:
#             #     orca_run.gradient = orca_run.engrad.gradient
#             # else:
#             #     orca_run.gradient = None
#             # # TODO implement hessian
#             logger.info(f"Done running ORCA calculation task_id: {task_id}")
#         else:
#            raise ValueError(f"orca.out file not found for task_id: {task_id}")
#     except Exception as e:
#         result_dict["error"] = str(e)
#         result_dict["status"] = "output analysis failed"
#         result_dict["task_status"] = TaskStatus.FAILED
#         logger.error(f"Error analyzing ORCA output: task_id: {task_id} {str(e)}")
#         return QMResult(**result_dict)

#     try:
#         if orca_run is not None:
#             logger.info(f"Final Structure for task_id: {task_id} {orca_run.final_structure}")
#             orca_result = QMResult.from_orca_output(orca_run, task_id)
#             logger.info(f"Done reading ORCA result task_id: {task_id}")
#         else:
#             logger.error(f"orca_run is none for task_id: {task_id}")
#             raise ValueError("orca_run is none")
#         orca_result.task_status = TaskStatus.COMPLETED
#         return orca_result
#     except Exception as e:
#         logger.error(f"Error reading ORCA result: task_id: {task_id} {str(e)}")
#         return QMResult(status="could not read result", task_status=TaskStatus.FAILED,
#                         error = str(e))

def make_water() -> Molecule:
    """
    Create a water molecule with 2 hydrogen and 1 oxygen
    """
    coords = [
        [0.0, 0.0, 0.05],  # O atom at origin
        [0.0, 0.757, 0.986],  # H atom 1
        [0.0, -0.757, 0.586]  # H atom 2
    ]

    # Define the atomic species
    species = ["O", "H", "H"]
    # Create the water molecule
    molecule = Molecule()
    for element, coord in zip(species, coords):
        atom = Atom.from_coords(element=element, coords=coord)
        molecule.add_atom(atom)
    return molecule

if __name__ == "__main__":
    # Don't create a new loop with asyncio.run, use an existing one
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Initialize context with this loop
    context.initialize()
    molecule = make_water()
    molecule = loop.run_until_complete(context.db.save(molecule))
    #depreciated
    # orca_input = QMInput(molecule=molecule, basis_set=BasisSet(basis_set="Def2_SVP"),
    #                      functional=Functional(functional="B3LYP", dispersion_correction=DispersionCorrection()), states=1, gradients=False)
    # #
    # # params = [orca_input.functional + " " + orca_input.basis]
    # # blocks = []
    # #
    # # orca_input_file = OrcaInput(orca_input.molecule, input_parameters=params, blocks=blocks)
    # # print(orca_input_file)
    # # sys.exit(1)

    # result = orca_v01(orca_input)
    # print("Result: ", result)
    # loop.close()

    # # # wf = Workflow()
    # # orca_node = Orca(orca_input)
    # # gmx_node = Gromacs(structureorca=orca_node.output.relaxed_structure)
    # # wf.add_nodes(orca_node, gmx_node)
    # # wf.run()
    # # wf = Workflow.from_json(input.json)
    # # wf.nodes.orca.input =
    # wf.nodes.orca.conda_env
    #
    # @node
    # def wf(qm_input: QMInput, **kwargs) -> OrcaResult:
    #     orca_result = orca(qm_input)
    #     gmx_result = gromacs(orca_result)
    #     return gmx_result